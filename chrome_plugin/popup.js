document.addEventListener("DOMContentLoaded", () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = "AIzaSyAeccguIJta5jSG09DFxjlfYep5WmJEsGE";
  const API_URL = "http://localhost:5000"; // Changed to local server

  // Get current tab URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const match = url.match(/v=([\w-]{11})/);

    // Update the status messages
    if (!match) {
      outputDiv.innerHTML = `<div class="error">Not a valid YouTube URL</div>`;
      return;
    }

    const videoId = match[1];
    outputDiv.innerHTML = `<div class="status">Video ID: ${videoId}<br>Fetching comments...</div>`;

    const comments = await fetchComments(videoId);
    if (!comments.length) {
      outputDiv.innerHTML += `<div class="error">No comments found or comments disabled</div>`;
      return;
    }

    outputDiv.innerHTML = `<div class="status">Fetched ${comments.length} comments.<br>Analyzing sentiment...</div>`;

    const predictions = await getPredictions(comments);
    if (!predictions) return;

    // Count sentiments
    const counts = { 0: 0, 1: 0 };
    predictions.forEach((p) => counts[p.sentiment]++);

    // Display summary
    outputDiv.innerHTML += `
      <div class="summary-box">
        <div class="section-title">Analysis Summary</div>
        <div class="stats">
          <div class="stat-item">
            <div class="stat-value">${comments.length}</div>
            <div class="stat-label">Total Comments</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">${counts[1]}</div>
            <div class="stat-label">Positive</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">${counts[0]}</div>
            <div class="stat-label">Negative</div>
          </div>
        </div>
      </div>
      <div id="chart"></div>
      <div id="wordcloud"></div>
      <div class="section-title">Sentiment Trends</div>
      <div id="trend-graph"></div>
      <div class="section-title">Top Comments</div>
      <ul class="comment-list">
        ${predictions
          .slice(0, 10)
          .map(
            (p) =>
              `<li class="comment-item">
                ${p.comment}
                <div class="sentiment-label ${p.sentiment ? 'positive' : 'negative'}">
                  ${p.sentiment_label}
                </div>
              </li>`
          )
          .join("")}
      </ul>
    `;

    // Fetch sentiment pie chart
    fetchImage(
      `${API_URL}/generate_chart`,
      { sentiment_counts: counts },
      "chart"
    );

    // Fetch wordcloud
    fetchImage(
      `${API_URL}/generate_wordcloud`,
      { comments: comments.map((c) => c.text) },
      "wordcloud"
    );
    // Add after your other chart generations
    fetchImage(
      `${API_URL}/generate_trend_graph`,
      { comments: predictions },
      "trend-graph"
    );
  });

  // ------------------- Helper Functions ------------------- //

  // Fetch comments from YouTube API (up to 200)
  async function fetchComments(videoId) {
    let comments = [];
    let token = "";
    try {
      while (comments.length < 200) {
        const url = `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100${
          token ? `&pageToken=${token}` : ""
        }&key=${API_KEY}`;
        const res = await fetch(url);
        const data = await res.json();

        if (data.error) {
          throw new Error(JSON.stringify(data.error));
        }

        if (data.items) {
          data.items.forEach((item) => {
            comments.push({
              text: item.snippet.topLevelComment.snippet.textOriginal,
              timestamp: item.snippet.topLevelComment.snippet.publishedAt,
            });
          });
        }

        token = data.nextPageToken || "";
        if (!token) break; // No more pages
      }
    } catch (err) {
      console.error("YouTube API error:", err);
      outputDiv.innerHTML += `<p>❌ Error fetching comments: ${err.message}</p>`;
    }
    return comments;
  }

  // Get sentiment predictions from Flask backend
  async function getPredictions(comments) {
    try {
      const res = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(
          errorData.error || `Server responded with status ${res.status}`
        );
      }

      const data = await res.json();
      return data;
    } catch (err) {
      console.error("Prediction error:", err);
      outputDiv.innerHTML += `<p>❌ Error fetching predictions: ${err.message}<br>Please ensure the API server is running at ${API_URL}</p>`;
      return null;
    }
  }

  // Fetch image (pie chart or wordcloud) from Flask backend
  async function fetchImage(url, payload, containerId) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Image fetch failed");

      const blob = await res.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement("img");
      img.src = imgURL;
      img.style.width = "100%";
      document.getElementById(containerId).appendChild(img);
    } catch (err) {
      console.error("Image fetch error:", err);
      outputDiv.innerHTML += `<p>❌ Error fetching image: ${err.message}</p>`;
    }
  }
});
