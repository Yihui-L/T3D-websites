(() => {
  "use strict";
  const chapters = [...document.querySelectorAll("[data-legacy-chapter]")];
  if (chapters.length) {
    const route = () => {
      const oldId = location.hash.slice(1);
      const chapter = chapters.find(({id}) => oldId === id || oldId === id + "-title");
      if (!chapter) return;
      const next = new URL(chapter.querySelector("a").href);
      next.search = location.search;
      next.hash = oldId;
      location.replace(next.href);
    };
    window.addEventListener("hashchange", route);
    route();
    return;
  }

  const form = document.querySelector(".manual-chapter .toolbar");
  if (!form) return;
  const field = form.querySelector("#search");
  const status = form.querySelector("#search-status");
  const collection = [...document.querySelectorAll("#chapter-change-items > li")].map(item => ({
    item, text: item.textContent.normalize("NFKC").toLowerCase()
  }));
  function filter() {
    const terms = field.value.normalize("NFKC").toLowerCase().trim().split(/\s+/u).filter(Boolean);
    let count = 0;
    for (const {item, text} of collection) {
      item.hidden = !terms.every(term => text.includes(term));
      if (!item.hidden) count++;
    }
    status.textContent = !terms.length ? "" : count ? "找到 " + count + " 条" : "本章没有匹配的改动。";
  }
  form.addEventListener("submit", event => event.preventDefault());
  field.addEventListener("input", filter);
  field.addEventListener("keydown", event => {
    if (event.key === "Escape" && field.value) {
      field.value = "";
      filter();
    }
  });
  form.hidden = false;
})();
