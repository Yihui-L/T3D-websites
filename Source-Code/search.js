(() => {
  "use strict";
  const form = document.querySelector(".toolbar");
  const field = document.querySelector("#search");
  const status = document.querySelector("#search-status");
  const groups = [...document.querySelectorAll(".topic")].map(topic => ({
    topic,
    title: topic.querySelector("h2").textContent.normalize("NFKC").toLowerCase(),
    list: [...topic.querySelectorAll(".changes > li")].map(item => ({
      item, text: item.textContent.normalize("NFKC").toLowerCase()
    }))
  }));
  const links = [...document.querySelectorAll(".sidebar nav a")];
  function filter() {
    const terms = field.value.normalize("NFKC").toLowerCase().trim().split(/\s+/u).filter(Boolean);
    let n = 0;
    for (const {topic, title, list} of groups) {
      let count = 0;
      for (const {item, text} of list) {
        item.hidden = !terms.every(word => (title + " " + text).includes(word));
        if (!item.hidden) count++;
      }
      topic.hidden = !!terms.length && !count;
      const link = links.find(a => a.hash === "#" + topic.id);
      if (link) link.parentElement.hidden = topic.hidden;
      n += count;
    }
    status.textContent = !terms.length ? "" : n ? "找到 " + n + " 条" : "没有匹配条目，请尝试其他关键词。";
  }
  form.addEventListener("submit", event => event.preventDefault());
  field.addEventListener("input", filter);
  field.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      field.value = "";
      filter();
    }
  });
})();
