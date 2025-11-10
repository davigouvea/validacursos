// Botões do Dashboard (gráficos)
document.addEventListener("click", async (ev) => {
  const id = ev.target.id;

  if (id === "btnBar" || id === "btnPie") {
    const kind = (id === "btnBar") ? "bar" : "pie";
    const col = document.getElementById("catCol").value;
    const { data, layout } = (await axios.post("/chart", { kind, col })).data;
    Plotly.newPlot("plot_cat", data, layout);
  }

  if (id === "btnScatter") {
    const x = document.getElementById("numX").value;
    const y = document.getElementById("numY").value;
    const color = document.getElementById("colorBy").value || null;
    const { data, layout } = (await axios.post("/chart", { kind:"scatter", x, y, color })).data;
    Plotly.newPlot("plot_scatter", data, layout);
  }

  if (id === "btnTrain") {
    const target = document.getElementById("target").value;
    const featsSel = document.getElementById("features");
    const features = [...featsSel.options].filter(o=>o.selected).map(o=>o.value).filter(f=>f!==target);
    const model = document.getElementById("model").value;
    if (features.length===0) { alert("Selecione ao menos uma feature (diferente do alvo)."); return; }
    const res = await axios.post("/train", { target, features, model });
    document.getElementById("trainOut").innerText = JSON.stringify(res.data, null, 2);

    // Monta inputs para predição
    const holder = document.getElementById("predictInputs");
    holder.innerHTML = "";
    features.forEach(f=>{
      const div = document.createElement("div");
      div.className = "mb-1";
      div.innerHTML = `<label class="form-label small">${f}</label><input class="form-control form-control-sm" data-feat="${f}">`;
      holder.appendChild(div);
    });
    holder.setAttribute("data-model", res.data.model_id);
  }

  if (id === "btnPredict") {
    const holder = document.getElementById("predictInputs");
    const model_id = holder.getAttribute("data-model");
    if (!model_id) { alert("Treine um modelo primeiro."); return; }
    const inputs = holder.querySelectorAll("[data-feat]");
    const payload = {};
    inputs.forEach(inp=>{
      const key = inp.getAttribute("data-feat");
      const val = inp.value;
      const num = Number(val);
      payload[key] = (val!=="" && !isNaN(num)) ? num : val;
    });
    axios.post("/predict", { model_id, payload }).then(res=>{
      document.getElementById("predOut").innerText = JSON.stringify(res.data, null, 2);
    });
  }
});
