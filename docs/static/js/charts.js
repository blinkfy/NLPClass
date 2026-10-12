import { api, node } from "./api.js";
import { renderModelInfo } from "./algorithm-view.js";

const $ = (id) => document.getElementById(id);
const charts = new Map();
const ink = "#14273e",
  teal = "#087f8c";
function chart(id, options) {
  if (!window.echarts) throw new Error("ECharts 资源未加载，请检查静态文件。");
  const element = $(id);
  element.classList.remove("empty-chart");
  if (!charts.has(id)) element.replaceChildren();
  const instance = charts.get(id) || echarts.init(element);
  charts.set(id, instance);
  instance.setOption(
    {
      backgroundColor: "#fff",
      textStyle: {
        fontFamily: "Microsoft YaHei, Segoe UI, sans-serif",
        color: ink,
      },
      animation: false,
      ...options,
    },
    true,
  );
}
export function resetCharts() {
  charts.forEach((instance) => instance.dispose());
  charts.clear();
  for (const id of [
    "start-chart",
    "transition-chart",
    "emission-chart",
    "metrics-chart",
    "confusion-chart",
  ]) {
    if (!$(id)) continue;
    $(id).classList.add("empty-chart");
    $(id).textContent = "尚无实验结果";
  }
  for (const id of ["model-metadata", "evaluation-info", "metrics-values"])
    $(id)?.replaceChildren();
  if ($("unknown-info")) $("unknown-info").textContent = "";
  if ($("ner-metrics")) $("ner-metrics").textContent = "尚无 NER 评价结果。";
}
export function resizeCharts() {
  charts.forEach((instance) => instance.resize());
}
export function exportChart(id, run) {
  if (!charts.has(id)) throw new Error("图表尚无数据，无法导出。");
  const anchor = document.createElement("a");
  anchor.href = charts
    .get(id)
    .getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#fff" });
  anchor.download = `${run}_${id}.png`;
  anchor.click();
}
function metadata(target, entries) {
  $(target).replaceChildren(
    ...entries.map(([label, value]) => {
      const item = node("span", label);
      item.append(node("strong", String(value)));
      return item;
    }),
  );
}
function heatmap(id, states, matrix, maximum, percent, predictedStates = states) {
  chart(id, {
    tooltip: {
      formatter: (value) =>
        `${states[value.data[1]]} → ${predictedStates[value.data[0]]}: ${percent ? value.data[2].toFixed(4) : value.data[2]}`,
    },
    grid: { left: 58, right: 30, top: 24, bottom: 70 },
    xAxis: {
      type: "category",
      data: predictedStates,
      name: percent ? "目标状态" : "预测",
      nameLocation: "middle",
      nameGap: 26,
      splitArea: { show: true },
    },
    yAxis: {
      type: "category",
      data: states,
      inverse: true,
      name: percent ? "源状态" : "Gold",
      splitArea: { show: true },
    },
    visualMap: {
      min: 0,
      max: maximum || 1,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      itemWidth: 10,
      itemHeight: 110,
      calculable: false,
      inRange: { color: ["#f0f8f8", "#087f8c"] },
    },
    series: [
      {
        type: "heatmap",
        data: matrix.flatMap((row, y) => row.map((value, x) => [x, y, value])),
        label: {
          show: true,
          formatter: (value) =>
            percent ? value.data[2].toFixed(3) : value.data[2],
        },
        emphasis: { itemStyle: { borderColor: ink, borderWidth: 1 } },
      },
    ],
  });
}
export async function loadParameters(
  run,
  topk,
  state,
  valid = () => true,
  modelName = "hmm1-smooth",
) {
  const value = await api(
    `/api/v1/models/hmm/parameters?run_id=${encodeURIComponent(run)}&topk=${topk}&model=${encodeURIComponent(modelName)}`,
  );
  if (!valid()) return;
  const model = value.parameters;
  renderModelInfo($("model-specific-parameters"), value, modelName, null, {trainingCurve: false});
  $("start-chart").closest(".charts-grid").classList.toggle("neural-charts", !!model.architecture);
  $("transition-chart").closest(".panel").hidden = !!model.architecture;
  $("start-chart").closest(".charts-grid").hidden = !model.start && !model.architecture;
  $("emission-chart").closest(".emission-panel").hidden = !model.start;
  $("emission-controls").hidden = !model.start;
  $("emission-title").textContent = model.architecture ? "BMES 发射分数" : !model.start ? "词典与路径说明" : "发射概率 B · Top-K";
  $("start-title").textContent = model.architecture ? "训练曲线" : "初始状态 π";
  $("transition-title").textContent = model.architecture ? "CRF 转移分数" : model.transition_tensor ? "二阶转移张量 · prev=B" : "状态转移 A";
  if (model.architecture) {
    metadata("model-metadata", [["结构", `Embedding ${model.architecture.embedding_dim} → BiLSTM ${model.architecture.hidden_size_per_direction}×2 → Linear 4 → CRF`], ["参数", model.parameter_count], ["词表", model.vocabulary_size], ["训练 epoch", model.training.best_epoch]]);
    const training = value.training_history?.selected_from_development || value.training_history;
    const history = training?.history || [];
    chart("start-chart", {
      tooltip: { trigger: "axis" }, legend: { data: ["train loss", "dev loss", "dev F1"] },
      xAxis: { type: "category", data: history.map((row) => row.epoch) },
      yAxis: [{ type: "value", name: "NLL" }, { type: "value", name: "F1", max: 1 }],
      series: [
        { name: "train loss", type: "line", data: history.map((row) => row.train_loss) },
        { name: "dev loss", type: "line", data: history.map((row) => row.dev_loss) },
        { name: "dev F1", type: "line", yAxisIndex: 1, data: history.map((row) => row.dev_f1) },
      ],
    });
    charts.get("transition-chart")?.dispose(); charts.delete("transition-chart");
    charts.get("emission-chart")?.dispose(); charts.delete("emission-chart");
    $("emission-chart").textContent = "上下文相关的 BMES emission 与解码路径见短文本算法过程。";
    $("unknown-info").textContent = "CRF transition 是可学习分数；非法 BMES 起止及转移受 mask 约束。";
    return;
  }
  if (!model.start) {
    metadata("model-metadata", [
      ["模型", model.model],
      ["词典大小", model.dictionary_size ?? model.vocabulary_size],
      ["最大词长", model.max_word_length ?? "—"],
      ["训练句数", model.sentence_count],
      ["一元词数", model.word_count],
      ["二元词对", model.bigram_types],
      ...(model.interpolation ? [["平滑", `λ=${model.interpolation}, α=${model.alpha}`]] : []),
      ["路径说明", modelName === "bigram" ? "词网 + 负对数代价 + 前驱" : "算法 trace 见 Run"],
    ]);
    for (const id of ["start-chart", "transition-chart", "emission-chart"]) {
      charts.get(id)?.dispose();
      charts.delete(id);
      if (!$(id)) continue;
    $(id).classList.add("empty-chart");
      $(id).textContent = "该模型不提供一阶 HMM 概率图；可查看 Run 中的过程 trace。";
    }
    if ($("unknown-info")) $("unknown-info").textContent = "";
    return;
  }
  metadata("model-metadata", [
    ["模型", model.model],
    ["训练句数", model.training.sentence_count],
    ["训练字符", model.training.character_count],
    ["字符词表", model.training.vocabulary_size],
    ["配置", `${model.config.smoothing} · α=${model.config.alpha}`],
    ...(model.transition_tensor ? [["二阶张量", "slice prev=B"]] : []),
  ]);
  chart("start-chart", {
    tooltip: { trigger: "axis" },
    grid: { left: 50, right: 20, top: 24, bottom: 30 },
    xAxis: { type: "category", data: model.states },
    yAxis: { type: "value", max: 1 },
    series: [
      {
        type: "bar",
        data: model.start,
        barWidth: 36,
        itemStyle: { color: teal },
        label: {
          show: true,
          position: "top",
          formatter: (p) => p.value.toFixed(3),
        },
      },
    ],
  });
  heatmap(
    "transition-chart",
    model.states,
    model.transition_tensor ? model.transition_tensor[0] : model.transition,
    1,
    true,
  );
  const entries = model.emission_topk[state];
  chart("emission-chart", {
    tooltip: { trigger: "axis" },
    grid: { left: 70, right: 70, top: 16, bottom: 50 },
    xAxis: {
      type: "value",
      name: "P(character | state)",
      nameLocation: "middle",
      nameGap: 28,
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: entries.map((x) => x.character),
      axisLabel: { interval: 0 },
    },
    series: [
      {
        type: "bar",
        data: entries.map((x) => x.probability),
        barMaxWidth: 18,
        itemStyle: { color: teal },
        label: {
          show: true,
          position: "right",
          formatter: (p) => p.value.toFixed(4),
        },
      },
    ],
  });
  $("unknown-info").textContent =
    `<UNK> · ${state}: ${model.unknown_probabilities[model.states.indexOf(state)].toFixed(6)} · 未知字符使用同一个归一化词表桶。`;
}
export async function loadComparison(valid = () => true) {
  const value = await api("/api/v1/experiments/1/comparison");
  if (!valid()) return;
  const target = $("comparison-table");
  target.replaceChildren();
  const table = node("table", "", "comparison-table");
  const head = node("tr");
  ["模型", "P", "R", "F1", "OOV-R", "IV-R", "Tag Acc.", "Coverage"].forEach((label) =>
    head.append(node("th", label)),
  );
  table.append(head);
  for (const row of value.rows) {
    const tr = node("tr");
    tr.append(node("td", row.model));
    for (const key of ["precision", "recall", "f1", "oov_recall", "iv_recall", "tag_accuracy", "coverage"])
      tr.append(node("td", row[key] == null ? "—" : `${(row[key] * 100).toFixed(2)}%${key === "tag_accuracy" && row.tag_accuracy_derived_from_saved_predictions ? " *" : ""}`));
    table.append(tr);
  }
  target.append(table);
  if (value.rows.some(row => row.tag_accuracy_derived_from_saved_predictions))
    target.append(node("p", "* 由冻结的词序列预测只读转换为 BMES 后派生，未重新执行模型推理。", "muted"));
}
export async function loadMetrics(run, valid = () => true) {
  const value = await api(
    `/api/v1/experiments/1/metrics?run_id=${encodeURIComponent(run)}`,
  );
  if (!valid()) return;
  const data = value.metrics.segmentation;
  if (data) {
    metadata("evaluation-info", [
      ["模型", value.run.model],
      ["数据集", value.run.dataset],
      ["评价句数", data.sentence_count],
      ["评价字符", data.character_count],
      ["OOV Gold", data.oov_support],
      ["IV Gold", data.iv_support],
    ]);
    const labels = [
      ["Tag Accuracy", "tag_accuracy"],
      ["Precision", "precision"],
      ["Recall", "recall"],
      ["F1", "f1"],
      ["OOV Recall", "oov_recall"],
      ["IV Recall", "iv_recall"],
    ];
    $("metrics-values").replaceChildren(
      ...labels.map(([label, key]) => {
        const item = node("div", "", "metric");
        item.append(
          node("span", label),
          node(
            "strong",
            data[key] === null ? "—" : `${(data[key] * 100).toFixed(2)}%`,
          ),
        );
        if (data[key] === null) item.append(node("small", "无对应 Gold 样本"));
        return item;
      }),
    );
    chart("metrics-chart", {
      tooltip: { trigger: "axis" },
      grid: { left: 48, right: 16, top: 28, bottom: 62 },
      xAxis: {
        type: "category",
        data: labels.map((x) => x[0]),
        axisLabel: { rotate: 25, fontSize: 10 },
      },
      yAxis: { type: "value", max: 1 },
      series: [
        {
          type: "bar",
          data: labels.map(([, key]) => data[key]),
          barMaxWidth: 30,
          itemStyle: { color: teal },
          label: {
            show: true,
            position: "top",
            formatter: (p) => `${(p.value * 100).toFixed(1)}%`,
          },
        },
      ],
    });
    heatmap(
      "confusion-chart",
      data.states,
      data.confusion_matrix,
      Math.max(...data.confusion_matrix.flat()),
      false,
      data.prediction_labels || data.states,
    );
  } else {
    metadata("evaluation-info", [
      ["模型", value.run.model],
      ["数据集", value.run.dataset],
      ["评价句数", value.run.sample_count],
    ]);
    $("metrics-values").textContent = "MSRA 无 CWS Gold；分词/BMES 仅作预测展示。CWS 正式指标请选择冻结 SIGHAN 2005 MSR Run。";
    for (const id of ["metrics-chart", "confusion-chart"]) {
      charts.get(id)?.dispose();
      charts.delete(id);
      if (!$(id)) continue;
    $(id).classList.add("empty-chart");
      $(id).textContent = "此数据集不计算 CWS Gold 指标";
    }
  }
  const ner = value.metrics.ner;
  $("ner-metrics").replaceChildren();
  if (!ner) {
    $("ner-metrics").textContent =
      "本 Run 未进行 NER 评价；需要人工 Gold 和可用的 HanLP 模型。";
    return;
  }
  const table = node("table"),
    head = node("tr");
  ["类别", "Precision", "Recall", "F1", "Gold 实体"].forEach((label) =>
    head.append(node("th", label)),
  );
  table.append(head);
  for (const label of ["PER", "LOC", "ORG", "Overall"]) {
    const row = node("tr");
    row.append(node("td", label));
    for (const key of ["precision", "recall", "f1"])
      row.append(node("td", `${(ner[label][key] * 100).toFixed(2)}%`));
    row.append(node("td", String(ner[label].support)));
    table.append(row);
  }
  const wrapper = node("div", "", "table-wrap");
  wrapper.append(table);
  $("ner-metrics").append(wrapper);
}
