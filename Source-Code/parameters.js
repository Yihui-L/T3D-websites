(() => {
  "use strict";
  const form = document.querySelector(".param-toolbar");
  const input = document.querySelector("#param-search");
  const status = document.querySelector("#param-status");
  if (!form || !input || !status) return;
  const normalize = text => text.normalize("NFKC").toLowerCase().trim();
  const registry = [...document.querySelectorAll(".param-group")].map(section => {
    const groupHeading = section.querySelector("h2");
    const group = normalize(groupHeading.textContent);
    let sub = "";
    const parts = [...section.children].flatMap(node => {
      if (node.matches("h3, h4")) sub = normalize(node.textContent);
      if (!node.matches(".table-scroll")) return [];
      const rows = [...node.querySelectorAll("tbody tr")].map(row => ({
        row, name: normalize(group + " " + sub + " " + row.textContent)
      }));
      return [{node, rows}];
    });
    return {section, parts, groupHeading};
  });
  form.hidden = false;
  form.addEventListener("submit", event => event.preventDefault());
  function filter() {
    const terms = normalize(input.value).split(/\s+/u).filter(Boolean);
    let sum = 0;
    for (const {section, parts, groupHeading} of registry) {
      let groupCount = 0;
      for (const {node, rows} of parts) {
        let matches = 0;
        for (const {row, name} of rows) {
          row.hidden = !terms.every(term => name.includes(term));
          if (!row.hidden) matches++;
        }
        node.hidden = !!terms.length && !matches;
        groupCount += matches;
      }
      section.hidden = !!terms.length && groupCount === 0;
      // Keep original group anchor navigation in sync with search.
      const link = [...document.querySelectorAll(".sidebar a")].find(a => decodeURIComponent(a.hash.slice(1)) === groupHeading.id);
      if (link) link.parentElement.hidden = section.hidden;
      // Keep contextual paragraphs/headings only when a group has results.
      sum += groupCount;
    }
    status.textContent = terms.length ? (sum ? "匹配 " + sum + " 条记录" : "没有匹配参数") : "";
  }
  input.addEventListener("input", filter);
  input.addEventListener("keydown", event => {
    if (event.key === "Escape") { input.value = ""; filter(); }
  });
})();
