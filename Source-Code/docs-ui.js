(() => {
  "use strict";
  const root = document.documentElement;
  const menu = document.querySelector("[data-doc-toggle]");
  const rail = document.querySelector("#doc-sidebar");
  if (menu && rail) {
    root.classList.add("doc-enhanced");
    menu.hidden = false;
    const close = focus => {
      rail.classList.remove("is-open");
      menu.setAttribute("aria-expanded", "false");
      if (focus) menu.focus();
    };
    menu.addEventListener("click", () => {
      const open = rail.classList.toggle("is-open");
      menu.setAttribute("aria-expanded", String(open));
      if (open) rail.querySelector("input")?.focus();
    });
    rail.addEventListener("click", event => { if (event.target.closest("a")) close(false); });
    document.addEventListener("keydown", event => {
      if (event.key === "Escape" && rail.classList.contains("is-open")) close(true);
    });
    const decode = hash => { try {return decodeURIComponent(hash);} catch {return hash;} };
    const update = () => {
      for (const link of rail.querySelectorAll('a[href^="#"]')) {
        if (link.hash && decode(link.hash) === decode(location.hash) && location.hash) {
          link.setAttribute("aria-current", "location");
        } else {
          link.removeAttribute("aria-current");
        }
      }
    };
    window.addEventListener("hashchange", update);
    update();
  }
  const form = document.querySelector("#doc-search");
  const items = document.querySelector("#doc-search-items");
  const status = document.querySelector("#doc-search-status");
  if (!form || !items || !status) return;
  const input = form.querySelector("input");
  const clean = value => value.normalize("NFKC").toLowerCase().trim();
  function search() {
    const query = input.value.trim();
    const terms = clean(query).split(/\s+/u).filter(Boolean);
    items.replaceChildren();
    const corpus = window.T3D_SEARCH_INDEX || [];
    if (!terms.length) {status.textContent = "输入关键词查找手册、参数和算例。"; return;}
    const matches = corpus.map(entry => {
      const title = clean(entry.title), text = clean(entry.text);
      return {entry, score: terms.every(term => title.includes(term) || text.includes(term))
        ? terms.reduce((n, term) => n + (title.includes(term) ? 3 : 1), 0) : 0};
    }).filter(match => match.score).sort((a,b) => b.score-a.score);
    status.textContent = matches.length ? "匹配 " + matches.length + " 条" + (matches.length > 40 ? "（显示前 40 条）" : "") : "没有匹配结果。";
    for (const {entry} of matches.slice(0, 40)) {
      const row = document.createElement("li"), link = document.createElement("a"), desc = document.createElement("p");
      link.href = entry.url;
      link.textContent = entry.title;
      const value = clean(entry.text);
      const offset = value.indexOf(terms[0]);
      const start = offset > 32 ? offset - 32 : 0;
      desc.textContent = (start ? "…" : "") + entry.text.slice(start, start + 130) + (entry.text.length > start + 130 ? "…" : "");
      row.append(link, desc); items.append(row);
    }
  }
  form.addEventListener("submit", e => e.preventDefault());
  input.addEventListener("input", search);
  input.value = new URLSearchParams(location.search).get("q") || "";
  search();
})();
