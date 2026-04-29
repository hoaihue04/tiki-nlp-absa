"use strict";
let stackedChart = null;
let radarChart   = null;

// ─── Aspect Vietnamese map ─────────────────────────────────────────────────
const ASPECT_VI = {
  "PRODUCT#QUALITY":      "Chất lượng tổng thể",
  "PRODUCT#MATERIAL":     "Chất liệu",
  "PRODUCT#COMFORT":      "Cảm giác sử dụng",
  "PRODUCT#SIZE":         "Kích thước / độ vừa vặn",
  "PRODUCT#DESIGN":       "Thiết kế & kiểu dáng",
  "PRODUCT#SAFETY":       "Mức độ an toàn",
  "PRODUCT#FUNCTION":     "Tính năng & công dụng",
  "PRODUCT#DURABILITY":   "Độ bền",
  "PRODUCT#VALUE":        "Giá trị so với giá tiền",

  "PRICE#AFFORDABILITY":  "Mức giá",
  "PRICE#DISCOUNT":       "Ưu đãi & khuyến mãi",

  "DELIVERY#SPEED":       "Tốc độ giao hàng",
  "DELIVERY#PACKAGING":   "Chất lượng đóng gói",
  "DELIVERY#ACCURACY":    "Độ chính xác đơn hàng",

  "SELLER#SERVICE":       "Chất lượng dịch vụ của shop",
  "SELLER#RESPONSIVENESS":"Mức độ phản hồi của shop",
  "SELLER#AUTHENTICITY":  "Độ tin cậy / chính hãng",
};

const ASPECT_ICON = {
  "PRODUCT#QUALITY":      "🧵",
  "PRODUCT#MATERIAL":     "🪡",
  "PRODUCT#COMFORT":      "😌",
  "PRODUCT#SIZE":         "📐",
  "PRODUCT#DESIGN":       "🎨",
  "PRODUCT#SAFETY":       "🛡️",
  "PRODUCT#FUNCTION":     "⚙️",
  "PRODUCT#DURABILITY":   "💪",
  "PRODUCT#VALUE":        "💰",
  "PRICE#AFFORDABILITY":  "🏷️",
  "PRICE#DISCOUNT":       "🎫",
  "DELIVERY#SPEED":       "🚚",
  "DELIVERY#PACKAGING":   "📦",
  "DELIVERY#ACCURACY":    "✅",
  "SELLER#SERVICE":       "🛒",
  "SELLER#RESPONSIVENESS":"💬",
  "SELLER#AUTHENTICITY":  "🔐",
};
// ─── Debug flag ────────────────────────────────────────────────────────────
const DEBUG = true;

function debugLog(...args) {
    if (DEBUG) {
        console.log("[DEBUG]", ...args);
    }
}

// ─── Image proxy helper ────────────────────────────────────────────────────
function proxyImg(url) {
  if (!url) return "";
  return `/api/proxy-image?url=${encodeURIComponent(url)}`;
}

function aspectVi(raw) {
  if (!raw) return "—";
  return ASPECT_VI[raw] || raw.replace(/.*#/, "");
}

function aspectIcon(raw) {
  return ASPECT_ICON[raw] || "📋";
}

// ─── Formatters ────────────────────────────────────────────────────────────
function fmtNum(v) {
  if (v == null || typeof v !== "number" || Number.isNaN(v)) return "—";
  return v.toLocaleString("vi-VN");
}
function fmtPct(v) {
  if (v == null || typeof v !== "number" || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}
function fmtScore(v, decimals = 3) {
  return (v != null && typeof v === "number" && !Number.isNaN(v))
    ? v.toFixed(decimals) : "—";
}
function safeFixed(v, d = 1) {
  return (v != null && !Number.isNaN(Number(v))) ? Number(v).toFixed(d) : "—";
}
function fmtPrice(v) {
  if (!v || isNaN(Number(v)) || Number(v) === 0) return null;
  return Number(v).toLocaleString("vi-VN") + " ₫";
}
function escHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ─── Progress ring ─────────────────────────────────────────────────────────
function setProgress(pct, msg, sub) {
  const CIRCUMFERENCE = 213.6;
  const circle = document.getElementById("ring-circle");
  if (!circle) return;
  const offset = CIRCUMFERENCE - (Math.min(pct, 100) / 100) * CIRCUMFERENCE;
  circle.style.strokeDashoffset = offset;
  document.getElementById("progress-pct").textContent = `${Math.round(pct)}%`;
  document.getElementById("progress-bar").style.width = `${Math.min(pct, 100)}%`;
  if (msg) document.getElementById("loading-step").textContent = msg;
  if (sub !== undefined && sub !== "") document.getElementById("loading-sub").textContent = sub;
}

// ─── Helpers: trích tên sản phẩm từ URL Tiki ──────────────────────────────
function extractNameFromUrl(url) {
  if (!url) return "";
  // https://tiki.vn/ten-san-pham-p123456.html
  const m = url.match(/tiki\.vn\/(.+?)-p\d+/);
  if (!m) return "";
  return m[1]
    .split("-")
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

// ─── Render: Product Hero ──────────────────────────────────────────────────
// ─── Render: Product Hero (FIXED - HIỂN THỊ ĐẦY ĐỦ) ─────────────────────────────────────────────────
function renderHeader(data) {
  const pi  = data.product_info || {};
  const adv = data.advice       || {};
  const m   = data.metrics      || {};

  // ── 1. Badge "Nên mua / Cân nhắc / Thận trọng" ──────────────────────────
  const advLabel = adv.label || "Cân nhắc";
  const advTone  = adv.tone  ||
    (advLabel === "Nên mua" ? "positive" :
     advLabel === "Thận trọng" ? "negative" : "neutral");

  const badgeEl = document.getElementById("advice-badge");
  if (badgeEl) {
    badgeEl.textContent = advLabel;
    badgeEl.className   = "product-badge " + advTone;
  }

  // ── 2. Tên sản phẩm ───────────────────────────────────────────────────────
  const inputUrl = (document.getElementById("product-url") || {}).value || "";
  let productName = pi.name && pi.name !== "nan" && pi.name !== "None" ? pi.name : "";
  if (!productName || productName === "—") {
    const m2 = inputUrl.match(/tiki\.vn\/(.+?)-p\d+/);
    if (m2) productName = m2[1].replace(/-/g, " ").replace(/\b\w/g, l => l.toUpperCase());
  }
  if (!productName || productName === "—") productName = "Sản phẩm #" + (pi.product_id || "");

  const nameEl = document.getElementById("product-name");
  if (nameEl) nameEl.textContent = productName;

  // ── 3. Product ID + meta chips (HIỂN THỊ ĐẦY ĐỦ) ────────────────────────────
  const chips = [];
  
  // Product ID - HIỂN THỊ RÕ RÀNG
  if (pi.product_id && pi.product_id !== "nan" && pi.product_id !== "None") {
    chips.push('<span class="meta-chip product-id">🔖 Mã SP: ' + escHtml(String(pi.product_id)) + '</span>');
  } else {
    chips.push('<span class="meta-chip product-id">🔖 Mã SP: Không xác định</span>');
  }
  
  // Brand name
  if (pi.brand_name && pi.brand_name !== "nan" && pi.brand_name !== "None" && pi.brand_name !== "") {
    chips.push('<span class="meta-chip brand">🏢 ' + escHtml(pi.brand_name) + '</span>');
  }
  
  // Seller name
  if (pi.seller_name && pi.seller_name !== "nan" && pi.seller_name !== "None" && pi.seller_name !== "") {
    chips.push('<span class="meta-chip">🏪 ' + escHtml(pi.seller_name) + '</span>');
  }
  
  // Official badge
  if (pi.seller_is_official || pi.is_official) {
    chips.push('<span class="meta-chip official">✓ Official Store</span>');
  }
  
  // Giá - HIỂN THỊ GIÁ RÕ RÀNG
  if (pi.price && Number(pi.price) > 0) {
    const priceFormatted = Number(pi.price).toLocaleString("vi-VN") + " ₫";
    chips.push('<span class="meta-chip price">💰 ' + priceFormatted + '</span>');
  }
  
  // Original price + discount
  if (pi.original_price && Number(pi.original_price) > Number(pi.price)) {
    const origFormatted = Number(pi.original_price).toLocaleString("vi-VN") + " ₫";
    const discount = Math.round((1 - Number(pi.price) / Number(pi.original_price)) * 100);
    chips.push('<span class="meta-chip original-price">~~ ' + origFormatted + ' ~~</span>');
    chips.push('<span class="meta-chip discount">-' + discount + '%</span>');
  }
  
  // Sold quantity
  if (pi.sold_quantity && Number(pi.sold_quantity) > 0) {
    const soldFormatted = Number(pi.sold_quantity).toLocaleString("vi-VN");
    chips.push('<span class="meta-chip sold">✓ Đã bán: ' + soldFormatted + '</span>');
  } else if (pi.sold_text && pi.sold_text !== "nan" && pi.sold_text !== "") {
    chips.push('<span class="meta-chip sold">✓ ' + escHtml(pi.sold_text) + '</span>');
  }
  
  // Rating average
  if (pi.rating_average && Number(pi.rating_average) > 0) {
    chips.push('<span class="meta-chip rating">⭐ ' + Number(pi.rating_average).toFixed(1) + ' sao</span>');
  }
  
  // Review count
  if (pi.review_count && Number(pi.review_count) > 0) {
    chips.push('<span class="meta-chip review-count">📝 ' + Number(pi.review_count).toLocaleString("vi-VN") + ' đánh giá</span>');
  }
  
  // Link to Tiki
  const linkHref = (pi.product_url && pi.product_url.startsWith("https")) ? pi.product_url : inputUrl;
  if (linkHref && linkHref.includes("tiki.vn")) {
    chips.push('<a class="meta-chip link" href="' + escHtml(linkHref) + '" target="_blank" rel="noopener">🔗 Xem trên Tiki</a>');
  }

  const metaEl = document.getElementById("product-meta");
  if (metaEl) {
    metaEl.innerHTML = chips.length ? chips.join("") : '<span style="color:var(--text-3)">⚠️ Không có thông tin chi tiết</span>';
  }

  // ── 4. Ảnh sản phẩm ──────────────────────────────────────────────────────
  const imgWrap = document.getElementById("product-img-wrap");
  if (imgWrap) {
    const rawUrl = pi.thumbnail_url || pi.image_url || "";
    if (rawUrl && rawUrl !== "nan" && rawUrl !== "") {
      imgWrap.innerHTML = '<img id="main-product-img" src="' + escHtml(proxyImg(rawUrl)) + '" alt="product" loading="lazy" onerror="this.parentElement.innerHTML=\'<span class=product-image-placeholder>🛍️</span>\'"/>';
    } else {
      imgWrap.innerHTML = '<span class="product-image-placeholder">🛍️</span>';
    }
  }

  // ── 5. Gallery ────────────────────────────────────────────────────────────
  const extraImgs = (pi.images || []).filter(img => img && img !== "nan" && img !== "").slice(1, 5);
  const galleryWrap = document.getElementById("product-gallery-wrap");
  if (galleryWrap && extraImgs.length) {
    galleryWrap.innerHTML = '<div class="product-gallery">' +
      extraImgs.map(u => '<div class="gallery-thumb" onclick="swapMainImg(this,\'' + escHtml(proxyImg(u)) + '\')"><img src="' + escHtml(proxyImg(u)) + '" loading="lazy"/></div>').join("") +
    '</div>';
  } else if (galleryWrap) {
    galleryWrap.innerHTML = "";
  }

  // ── 6. Short description (nếu có) ─────────────────────────────────────────
  const shortDesc = pi.short_description || pi.description || "";
  const shortDescEl = document.getElementById("product-short-desc");
  if (shortDescEl && shortDesc && shortDesc !== "nan" && shortDesc !== "") {
    shortDescEl.style.display = "block";
    shortDescEl.innerHTML = escHtml(shortDesc.substring(0, 200)) + (shortDesc.length > 200 ? "..." : "");
  } else if (shortDescEl) {
    shortDescEl.style.display = "none";
  }

  // ── 7. Giá + điểm ABSA (HIỂN THỊ RÕ RÀNG) ───────────────────────────────────
  const price   = Number(pi.price) || 0;
  const origPx  = Number(pi.original_price) || 0;
  const disc    = (origPx > price && origPx > 0 && price > 0) ? Math.round((1 - price / origPx) * 100) : null;
  const absa5   = ((m.absa_score || 0) * 5).toFixed(1);
  const rating  = pi.rating_average ? Number(pi.rating_average).toFixed(1) : null;
  const soldTxt = pi.sold_text || (pi.sold_quantity > 0 ? Number(pi.sold_quantity).toLocaleString("vi-VN") : "");
  const scoreColor = advTone === "positive" ? "var(--green)" : advTone === "negative" ? "var(--red)" : "var(--amber)";

  const scoresEl = document.getElementById("product-scores");
  if (scoresEl) {
    let priceHtml = '';
    if (price > 0) {
      priceHtml = `
        <div class="price-current">
          <span class="price-value">${price.toLocaleString("vi-VN")}đ</span>
          ${disc ? '<span class="discount-badge">-' + disc + '%</span>' : ''}
        </div>
      `;
      if (origPx > price && origPx > 0) {
        priceHtml += '<div class="price-original">' + origPx.toLocaleString("vi-VN") + 'đ</div>';
      }
    }
    
    scoresEl.innerHTML = `
      <div class="product-price-panel">
        ${priceHtml}
        ${soldTxt ? '<div class="sold-count">✓ Đã bán: ' + soldTxt + '</div>' : ''}
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--line)">
          <div style="font-size:11px;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px">Điểm ABSA</div>
          <div style="font-size:26px;font-weight:800;color:${scoreColor}">${absa5}<span style="font-size:14px;color:var(--text-3)">/5</span></div>
          ${rating ? '<div style="font-size:12px;color:var(--text-3);margin-top:4px">⭐ ' + rating + ' sao (Tiki)</div>' : ''}
        </div>
      </div>
    `;
  }
}
function swapMainImg(thumbEl, newSrc) {
  const mainImg = document.getElementById("main-product-img");
  if (mainImg) mainImg.src = newSrc;
  document.querySelectorAll(".gallery-thumb").forEach(t => t.classList.remove("active"));
  thumbEl.classList.add("active");
}

// ─── Render: Metrics Strip ─────────────────────────────────────────────────
function renderMetrics(data) {
  const m       = data.metrics || {};
  const opinions = data.opinion_table || [];

  let posCount = 0, neuCount = 0, negCount = 0;
  if (opinions.length) {
    opinions.forEach(r => {
      if (r.sentiment === "positive") posCount += (r.count || 1);
      else if (r.sentiment === "negative") negCount += (r.count || 1);
      else neuCount += (r.count || 1);
    });
  }
  const total  = posCount + neuCount + negCount || 1;
  const posPct = m.positive_ratio != null
    ? (m.positive_ratio * 100).toFixed(1)
    : ((posCount / total) * 100).toFixed(1);
  const neuPct = m.neutral_ratio != null
    ? (m.neutral_ratio  * 100).toFixed(1)
    : ((neuCount / total) * 100).toFixed(1);
  const negPct = m.negative_ratio != null
    ? (m.negative_ratio * 100).toFixed(1)
    : ((negCount / total) * 100).toFixed(1);

  const METRICS_DEF = [
    { val: fmtNum(m.total_reviews_used),    label: "Tổng review",  accent: "#2563eb" },
    { val: fmtNum(m.num_aspects_mentioned), label: "Khía cạnh",    accent: "#0891b2" },
    { val: posPct + "%",                    label: "😊 Tích cực",  accent: "#059669" },
    { val: neuPct + "%",                    label: "😐 Trung lập", accent: "#d97706" },
    { val: negPct + "%",                    label: "😞 Tiêu cực",  accent: "#dc2626" },
  ];

  document.getElementById("metrics-grid").innerHTML = METRICS_DEF
    .map(({ val, label, accent }, i) => `
      <div class="metric-card" style="--accent:${accent}; animation-delay:${i * 60}ms">
        <div class="metric-label">${label}</div>
        <div class="metric-val">${val}</div>
      </div>
    `).join("");
}

// ─── Render: Charts ────────────────────────────────────────────────────────
const CHART_DEFAULTS = {
  color: "#64748b",
  grid:  { color: "rgba(0,0,0,0.05)", drawTicks: false },
  font:  { family: "'Be Vietnam Pro', sans-serif", size: 11 },
};

function renderCharts(data) {
  const aspect   = data.aspect || {};
  const rows     = (aspect.table || []);
  const radar    = (aspect.radar || []);
  const top1Scores = data.top1_aspect_scores || {};

  const mentionedRows  = rows.filter(r => (r.positive || 0) + (r.neutral || 0) + (r.negative || 0) > 0);
  const mentionedRadar = radar.filter(r => r.score != null && r.score > 0);

  const radarLabels      = mentionedRadar.map(r => aspectVi(r.aspect));
  const currentScores    = mentionedRadar.map(r => r.score || 0.5);
  const top1ScoresArray  = mentionedRadar.map(r => {
    const score = top1Scores[r.aspect];
    return score !== undefined ? score : 0.5;
  });

  if (stackedChart) { stackedChart.destroy(); stackedChart = null; }
  if (radarChart)   { radarChart.destroy();   radarChart   = null; }

  // Stacked bar chart
  const ctx1 = document.getElementById("aspectStackedChart").getContext("2d");
  stackedChart = new Chart(ctx1, {
    type: "bar",
    data: {
      labels: mentionedRows.map(r => aspectVi(r.aspect)),
      datasets: [
        { label: "Tích cực", data: mentionedRows.map(r => r.positive || 0),
          backgroundColor: "rgba(5,150,105,.7)",  borderRadius: 3 },
        { label: "Trung lập", data: mentionedRows.map(r => r.neutral  || 0),
          backgroundColor: "rgba(217,119,6,.65)", borderRadius: 3 },
        { label: "Tiêu cực", data: mentionedRows.map(r => r.negative  || 0),
          backgroundColor: "rgba(220,38,38,.65)", borderRadius: 3 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: CHART_DEFAULTS.color, font: CHART_DEFAULTS.font,
                    boxWidth: 10, boxHeight: 10, padding: 16 },
        },
        tooltip: { mode: "index", intersect: false },
      },
      scales: {
        x: { stacked: true, ticks: { color: CHART_DEFAULTS.color, font: CHART_DEFAULTS.font }, grid: CHART_DEFAULTS.grid },
        y: { stacked: true, ticks: { color: CHART_DEFAULTS.color, font: CHART_DEFAULTS.font }, grid: CHART_DEFAULTS.grid },
      },
    },
  });

  // Radar chart
  const ctx2 = document.getElementById("aspectRadarChart").getContext("2d");
  const top1Name       = data.recommendations?.top_products?.[0]?.name || "Top 1";
  const currentName    = data.product_info?.name || "Sản phẩm này";
  const shortTop1Name  = top1Name.length    > 25 ? top1Name.substring(0, 22)    + "..." : top1Name;
  const shortCurrName  = currentName.length > 25 ? currentName.substring(0, 22) + "..." : currentName;

  radarChart = new Chart(ctx2, {
    type: "radar",
    data: {
      labels: radarLabels,
      datasets: [
        {
          label: `📊 ${shortCurrName}`,
          data: currentScores,
          borderColor: "#2563eb",
          backgroundColor: "rgba(37,99,235,.08)",
          pointBackgroundColor: "#2563eb",
          pointBorderColor: "#fff",
          pointRadius: 4,
          borderWidth: 2.5,
        },
        {
          label: `🏆 ${shortTop1Name} (Top gợi ý)`,
          data: top1ScoresArray,
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245,158,11,.08)",
          pointBackgroundColor: "#f59e0b",
          pointBorderColor: "#fff",
          pointRadius: 4,
          borderWidth: 2.5,
          borderDash: [5, 5],
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: CHART_DEFAULTS.color,
                    font: { size: 10, family: CHART_DEFAULTS.font.family },
                    usePointStyle: true, boxWidth: 8 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) =>
              `${ctx.dataset.label || ""}: ${(ctx.raw * 5).toFixed(1)}/5 điểm`,
          },
        },
      },
      scales: {
        r: {
          min: 0, max: 1,
          ticks: {
            color: CHART_DEFAULTS.color, font: CHART_DEFAULTS.font,
            backdropColor: "transparent", stepSize: 0.25,
            callback: (val) => (val * 5).toFixed(0),
          },
          grid:        { color: "rgba(0,0,0,0.07)" },
          pointLabels: { color: CHART_DEFAULTS.color, font: { ...CHART_DEFAULTS.font, size: 10 } },
          angleLines:  { color: "rgba(0,0,0,0.06)" },
        },
      },
    },
  });
}

// ─── Render: Strengths / Weaknesses ───────────────────────────────────────
function renderSW(data) {
  const aspect = data.aspect || {};

  function swHtml(items, type) {
    if (!items || !items.length) {
      return `<li style="color:var(--text-3);padding:12px">Không có dữ liệu</li>`;
    }
    return items.map(r => {
      const score        = r.avg_score_0_1 ?? r.score ?? 0;
      const scoreSign    = type === "positive" ? "+" : "-";
      const scoreDisplay = `${scoreSign}${(score * 5).toFixed(1)}`;

      // FIX 2: Chọn đúng quote theo type (positive/negative)
      let quoteText = "";
      let likeCount = 0;

      if (type === "positive") {
        quoteText = r.best_positive_quote || "";
        likeCount = r.best_positive_quote_count || 0;
        // Không fallback sang negative quote — thà để trống còn hơn hiển thị sai
      } else {
        quoteText = r.best_negative_quote || "";
        likeCount = r.best_negative_quote_count || 0;
        // Không fallback sang positive quote — thà để trống còn hơn hiển thị sai
      }

      // Làm sạch quote: loại bỏ nan/None
      if (!quoteText || quoteText === "nan" || quoteText === "None") quoteText = "";

      let likeHtml = "";
      if (quoteText) {
        if (likeCount > 0) {
          likeHtml = `<span class="like-count">👍 ${likeCount} người thấy hữu ích</span>`;
        } else {
          likeHtml = `<span class="like-count" style="background:var(--blue-dim);color:var(--blue)">⭐ Độ tin cậy cao</span>`;
        }
      }

      // Thông tin bổ sung: số lượt negative/positive
      const negCount = r.negative || 0;
      const posCount = r.positive || 0;
      const subInfo  = type === "negative"
        ? (negCount > 0 ? `😞 ${negCount} đánh giá tiêu cực · ` : "")
        : (posCount > 0 ? `😊 ${posCount} đánh giá tích cực · ` : "");

      return `
        <li class="sw-item">
          <div class="sw-item-header">
            <span class="sw-aspect">${aspectIcon(r.aspect)} ${aspectVi(r.aspect)}</span>
            <span class="sw-score-badge">${scoreDisplay}</span>
          </div>
          ${quoteText
            ? `<div class="sw-quote">
                 <span class="quote-icon">💬</span>
                 "${escHtml(quoteText)}"
               </div>
               <div class="sw-meta">${likeHtml}</div>`
            : `<div class="sw-quote" style="font-style:normal;color:var(--text-3);font-size:12px">
                 📝 Chưa trích xuất được bình luận tiêu biểu
               </div>`
          }
          <div class="sw-conf">${subInfo}📊 ${r.mentions || 0} lượt nhắc đến</div>
        </li>
      `;
    }).join("");
  }

  const strengths  = aspect.strengths  || [];
  // FIX 3: backend đã lọc đúng — không filter lại ở frontend
  const weaknesses = aspect.weaknesses || [];

  document.getElementById("strength-list").innerHTML  = swHtml(strengths,  "positive");
  document.getElementById("weakness-list").innerHTML  = weaknesses.length
    ? swHtml(weaknesses, "negative")
    : `<li style="color:var(--text-3);padding:12px;font-size:13px">✅ Không tìm thấy điểm yếu đáng kể</li>`;
}

// ─── Render: Opinion Table ─────────────────────────────────────────────────
function sentLabel(s) {
  return s === "positive" ? "Tích cực" : s === "negative" ? "Tiêu cực" : "Trung lập";
}

function renderOpinion(data) {
  const table = (data.opinion_table || []).slice(0, 30);
  if (!table.length) {
    document.getElementById("opinion-body").innerHTML =
      `<tr><td colspan="6" style="color:var(--text-3);text-align:center;padding:20px">Không có dữ liệu</td></tr>`;
    return;
  }

  document.getElementById("opinion-body").innerHTML = table.map((r, i) => {
    const conf         = ((r.confidence || 0) * 100).toFixed(0);
    const cnt          = r.count ?? r.cnt ?? "—";
    const aspectCode   = r.aspect || "—";
    const aspectMeaning = aspectVi(aspectCode);
    const aspectIconChar = aspectIcon(aspectCode);

    return `
      <tr>
        <td>${String(i + 1).padStart(2, "0")}</td>
        <td>
          <div class="aspect-code">${aspectIconChar} ${aspectCode}</div>
        </td>
        <td>
          <div class="aspect-meaning">${aspectMeaning}</div>
        </td>
        <td><span class="sentiment-tag ${r.sentiment}">${sentLabel(r.sentiment)}</span></td>
        <td>
          <div class="score-bar-wrap">
            <div class="score-bar">
              <div class="score-bar-fill" style="width:${conf}%"></div>
            </div>
            <span style="font-family:var(--font-mono);font-size:11px;color:var(--text-3);min-width:32px">
              ${conf}%
            </span>
          </div>
        </td>
        <td style="font-family:var(--font-mono);color:var(--text-3);text-align:center">${cnt}</td>
      </tr>
    `;
  }).join("");
}

// ─── Render: Representative Reviews ───────────────────────────────────────
const REVIEW_AVATARS = {
  positive: ["😊", "🥰", "😄"],
  negative: ["😟", "😤", "😞"],
  neutral:  ["🤔", "😐", "🙂"],
};

function renderReviews(data) {
  const reviews = data.representative_reviews || [];
  const types   = ["Tích cực nhất", "Tiêu cực nhất", "Ngẫu nhiên"];

  if (!reviews.length) {
    document.getElementById("review-cards").innerHTML =
      `<p style="color:var(--text-3)">Không có đánh giá tiêu biểu.</p>`;
    return;
  }

  document.getElementById("review-cards").innerHTML = reviews.map((r, i) => {
    const sentiment   = r.label || "neutral";
    const starCount   = r.rating || (sentiment === "positive" ? 5 : sentiment === "negative" ? 2 : 3);
    const stars = Array.from({ length: 5 }, (_, si) =>
      `<span class="star ${si < starCount ? "on" : "off"}">★</span>`
    ).join("");
    const avatarEmoji = (REVIEW_AVATARS[sentiment] || REVIEW_AVATARS.neutral)[i % 3];
    const reviewText  = r.text && r.text !== "nan" ? r.text : "Không có nội dung đánh giá.";

    return `
      <article class="review-card ${sentiment}">
        <div class="review-card-header">
          <div class="review-avatar">${avatarEmoji}</div>
          <div class="review-meta">
            <div class="review-type">${types[i] || "Đánh giá"}  ·  ${sentLabel(sentiment)}</div>
            <div class="review-stars">${stars}</div>
          </div>
        </div>
        <div class="review-text">${escHtml(reviewText)}</div>
      </article>
    `;
  }).join("");
}

// ─── Render: Recommendations ───────────────────────────────────────────────
// ─── Render: Recommendations (dùng search Tiki thay vì link trực tiếp) ──────────
function renderRecommendations(data) {
  const rec = data.recommendations || {};
  
  // Ẩn phần hiển thị weights
  const weightsEl = document.getElementById("rec-weights");
  if (weightsEl) {
    weightsEl.style.display = "none";
  }

  const products = (rec.top_products || []).slice(0, 5);
  const container = document.getElementById("recommend-grid");

  if (!products.length) {
    container.innerHTML = `<p style="color:var(--text-3)">Không tìm thấy sản phẩm gợi ý cùng danh mục.</p>`;
    return;
  }

  container.innerHTML = products.map((r, i) => {
    // CÁCH CŨ: Dùng tìm kiếm trên Tiki thay vì link trực tiếp
    const searchQuery = encodeURIComponent(r.name || "");
    const searchUrl = `https://tiki.vn/search?q=${searchQuery}`;
    
    const imgSrc = r.thumbnail_url || "";
    const proxiedImg = imgSrc ? proxyImg(imgSrc) : "";
    const price = fmtPrice(r.price);

    return `
      <a class="rec-card ${i === 0 ? "rec-rank-1" : ""}" 
         href="${escHtml(searchUrl)}" 
         target="_blank" 
         rel="noopener noreferrer">
        <div class="rec-card-img">
          ${proxiedImg
            ? `<img src="${escHtml(proxiedImg)}" alt="${escHtml(r.name || "")}" loading="lazy" onerror="this.parentElement.innerHTML='🛍️'"/>`
            : "🛍️"
          }
        </div>
        <div class="rec-card-body">
          <div class="rec-rank">${i === 0 ? "🥇 Gợi ý #1" : `✨ Gợi ý #${i + 1}`}</div>
          <div class="rec-name">${escHtml(r.name || "—")}</div>
          ${price ? `<div class="rec-price">${price}</div>` : ""}
          <div class="rec-link-hint">🔍 Tìm kiếm trên Tiki</div>
        </div>
      </a>
    `;
  }).join("");
}
// Thêm hàm mới để hiển thị recommendation banner
// Thêm hàm này vào app.js nếu chưa có
function renderRecommendationBanner(data) {
  const advice  = data.advice  || {};
  const metrics = data.metrics || {};
  const absaScore = metrics.absa_score || 0;

  const banner     = document.getElementById("rec-banner");
  const recMessage = document.getElementById("rec-message");
  const recScore   = document.getElementById("rec-score");
  if (!banner || !recMessage || !recScore) return;

  const label = advice.label || "Cân nhắc";
  const tone  = advice.tone  ||
    (label === "Nên mua" ? "positive" : label === "Thận trọng" ? "negative" : "neutral");

  const CONFIG = {
    positive: {
      icon: "✅",
      msg:  "Sản phẩm được đánh giá <strong>rất tốt</strong>! Chất lượng vượt trội so với phần lớn sản phẩm cùng phân khúc.",
      bg:   "linear-gradient(135deg, rgba(5,150,105,.12) 0%, var(--surface) 100%)",
      border: "rgba(5,150,105,.4)",
      color:  "var(--green)",
      badgeBg: "#059669",
    },
    neutral: {
      icon: "⚠️",
      msg:  "Sản phẩm ở mức <strong>trung bình</strong>. Nên xem xét kỹ các đánh giá trước khi quyết định mua.",
      bg:   "linear-gradient(135deg, rgba(217,119,6,.1) 0%, var(--surface) 100%)",
      border: "rgba(217,119,6,.4)",
      color:  "var(--amber)",
      badgeBg: "#d97706",
    },
    negative: {
      icon: "❌",
      msg:  "Sản phẩm có <strong>nhiều đánh giá tiêu cực</strong>. Nên cân nhắc các sản phẩm thay thế được gợi ý bên dưới.",
      bg:   "linear-gradient(135deg, rgba(220,38,38,.1) 0%, var(--surface) 100%)",
      border: "rgba(220,38,38,.4)",
      color:  "var(--red)",
      badgeBg: "#dc2626",
    },
  };

  const cfg = CONFIG[tone] || CONFIG.neutral;

  banner.style.cssText = `
    display: flex;
    background: ${cfg.bg};
    border-color: ${cfg.border};
  `;

  const badgeEl  = banner.querySelector(".rec-badge");
  const iconEl   = banner.querySelector(".rec-icon");
  const textEl   = banner.querySelector(".rec-text");
  if (badgeEl)  badgeEl.style.background = cfg.badgeBg;
  if (iconEl)   iconEl.textContent = cfg.icon;
  if (textEl)   textEl.textContent = label.toUpperCase();

  recMessage.style.color = cfg.color;
  recMessage.innerHTML   = cfg.msg;

  const score5 = (absaScore * 5).toFixed(1);
  recScore.style.color = cfg.color;
  recScore.innerHTML   = `${score5}<span style="font-size:14px;opacity:.7">/5 ⭐</span>`;
}

// ─── Analyze with SSE progress ─────────────────────────────────────────────
async function analyze() {
  const url = document.getElementById("product-url").value.trim();
  if (!url) { alert("Vui lòng nhập link sản phẩm Tiki"); return; }

  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;

  document.getElementById("loading-overlay").classList.remove("hidden");
  document.getElementById("result-root").classList.add("hidden");
  setProgress(0, "Gửi yêu cầu tới server...", "Đang kết nối");

  let es = null;

  try {
    const startResp = await fetch("/api/analyze/start", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ product_url: url }),
    });

    if (!startResp.ok) {
      let errMsg = "Không thể bắt đầu phân tích";
      try { errMsg = (await startResp.json()).detail || errMsg; } catch (_) {}
      throw new Error(errMsg);
    }

    const { job_id } = await startResp.json();

    await new Promise((resolve, reject) => {
      es = new EventSource(`/api/analyze/progress/${job_id}`);

      const timeout = setTimeout(() => {
        es.close();
        reject(new Error("Phân tích quá thời gian (5 phút). Vui lòng thử lại."));
      }, 5 * 60 * 1000);

      es.onmessage = (event) => {
        let payload;
        try { payload = JSON.parse(event.data); } catch (_) { return; }

        setProgress(payload.progress ?? 0, payload.message ?? "", "");

        if (payload.status === "done") {
          clearTimeout(timeout);
          es.close();
          renderAll(payload.data);
          resolve();
        } else if (payload.status === "error") {
          clearTimeout(timeout);
          es.close();
          reject(new Error(payload.error || "Lỗi không xác định từ server"));
        }
      };

      es.onerror = () => {
        clearTimeout(timeout);
        es.close();
        reject(new Error("Mất kết nối SSE với server. Vui lòng kiểm tra server đang chạy."));
      };
    });

  } catch (err) {
    alert(`Lỗi: ${err.message}`);
  } finally {
    if (es) { try { es.close(); } catch (_) {} }
    document.getElementById("loading-overlay").classList.add("hidden");
    btn.disabled = false;
  }
}

// ─── Event Listeners ────────────────────────────────────────────────────────
document.getElementById("analyze-btn").addEventListener("click", analyze);
document.getElementById("product-url").addEventListener("keydown", (e) => {
  if (e.key === "Enter") analyze();
});

// Cập nhật hàm renderAll để gọi banner
function renderAll(data) {
  if (!data) { 
    console.error("renderAll: data is null"); 
    return; 
  }
  
  debugLog("=== RENDER ALL START ===");
  debugLog("Product info:", data.product_info);
  debugLog("Metrics:", data.metrics);
  debugLog("Aspect table length:", data.aspect?.table?.length);
  debugLog("Recommendations:", data.recommendations?.top_products?.length);
  
  try { renderHeader(data); }          catch (e) { console.error("renderHeader ERROR:", e); }
  try { renderMetrics(data); }         catch (e) { console.error("renderMetrics ERROR:", e); }
  try { renderCharts(data); }          catch (e) { console.error("renderCharts ERROR:", e); }
  try { renderSW(data); }              catch (e) { console.error("renderSW ERROR:", e); }
  try { renderOpinion(data); }         catch (e) { console.error("renderOpinion ERROR:", e); }
  try { renderReviews(data); }         catch (e) { console.error("renderReviews ERROR:", e); }
  try { renderRecommendations(data); } catch (e) { console.error("renderRecommendations ERROR:", e); }
  try { renderRecommendationBanner(data); } catch (e) { console.error("renderRecommendationBanner ERROR:", e); }

  document.getElementById("result-root").classList.remove("hidden");
  debugLog("=== RENDER ALL COMPLETE ===");
}