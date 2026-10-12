// 仅执行三维显示、视角旋转和拾取；SVD 坐标由服务提供。
export function boundedTooltipPosition(point, content, view) {
  const gap = 12;
  const padding = 8;
  const [width, height] = content;
  const rightRoom = view[0] - point[0] - gap;
  const leftRoom = point[0] - gap;
  let x = rightRoom >= width || rightRoom >= leftRoom ? point[0] + gap : point[0] - width - gap;
  let y = point[1] + gap;
  if (y + height > view[1] - padding) y = point[1] - height - gap;
  x = Math.max(padding, Math.min(x, view[0] - width - padding));
  y = Math.max(padding, Math.min(y, view[1] - height - padding));
  return [x, y];
}

export class Scatter3D {
  constructor(element, tooltipHTML, onSelect) {
    this.element = element;
    this.tooltipHTML = tooltipHTML;
    this.onSelect = onSelect;
    this.yaw = .65; this.pitch = -.35; this.zoom = 1;
    this.canvas = document.createElement('canvas');
    this.canvas.setAttribute('aria-label', '三维文档投影，拖动旋转，滚轮缩放');
    this.canvas.tabIndex = 0;
    this.tip = document.createElement('div');
    this.tip.className = 'clustering-point-tooltip'; this.tip.hidden = true;
    this.legend = document.createElement('div'); this.legend.className = 'clustering-3d-legend';
    this.reset = document.createElement('button'); this.reset.type = 'button'; this.reset.textContent = '重置视角';
    this.reset.className = 'clustering-3d-reset';
    element.replaceChildren(this.legend, this.canvas, this.reset, this.tip);
    this.abort = new AbortController();
    const options = { signal: this.abort.signal };
    this.reset.addEventListener('click', () => { this.yaw = .65; this.pitch = -.35; this.zoom = 1; this.draw(); }, options);
    this.canvas.addEventListener('pointerdown', (event) => {
      this.drag = [event.clientX, event.clientY]; this.moved = false;
      this.canvas.setPointerCapture(event.pointerId); this.tip.hidden = true;
    }, options);
    this.canvas.addEventListener('pointermove', (event) => {
      if (this.drag) {
        const dx = event.clientX - this.drag[0], dy = event.clientY - this.drag[1];
        this.moved ||= Math.abs(dx) + Math.abs(dy) > 2;
        this.yaw += dx * .008; this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + dy * .008));
        this.drag = [event.clientX, event.clientY]; this.draw();
      } else this.hover(event);
    }, options);
    this.canvas.addEventListener('pointerup', (event) => { this.drag = null; if (!this.moved) { this.hover(event); this.onSelect?.(this.pick(event)); } }, options);
    this.canvas.addEventListener('pointercancel', () => { this.drag = null; this.tip.hidden = true; }, options);
    this.canvas.addEventListener('pointerleave', () => { this.tip.hidden = true; }, options);
    this.canvas.addEventListener('wheel', (event) => {
      event.preventDefault(); this.zoom = Math.max(.55, Math.min(2.8, this.zoom * Math.exp(-event.deltaY * .001)));
      this.tip.hidden = true; this.draw();
    }, { ...options, passive: false });
    this.canvas.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '-'].includes(event.key)) return;
      event.preventDefault();
      if (event.key === 'ArrowLeft') this.yaw -= .1;
      if (event.key === 'ArrowRight') this.yaw += .1;
      if (event.key === 'ArrowUp') this.pitch -= .1;
      if (event.key === 'ArrowDown') this.pitch += .1;
      if (event.key === '+') this.zoom = Math.min(2.8, this.zoom * 1.1);
      if (event.key === '-') this.zoom = Math.max(.55, this.zoom / 1.1);
      this.tip.hidden = true; this.draw();
    }, options);
    this.observer = new ResizeObserver(() => { this.tip.hidden = true; this.draw(); }); this.observer.observe(element);
  }

  setData(points, groups, options = {}) {
    this.points = points; this.groups = groups;
    this.options = options;
    this.canvas.dataset.pointCount = String(points.length);
    this.canvas.dataset.centroidCount = String((options.centers || []).length);
    this.tip.hidden = true;
    this.legend.replaceChildren(...groups.map((group) => {
      const label = document.createElement('span'); label.textContent = group.label;
      const dot = document.createElement('i'); dot.style.background = group.color; label.prepend(dot);
      return label;
    }));
    this.draw();
  }

  draw() {
    if (!this.points || !this.element.clientWidth || !this.element.clientHeight) return;
    const width = this.element.clientWidth, height = this.element.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(width * dpr); this.canvas.height = Math.round(height * dpr);
    const ctx = this.canvas.getContext('2d'); ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);
    const scale = Math.min(width * .25, (height - 80) * .34) * this.zoom;
    const center = this.bounds?.center || [0, 0, 0], radius = this.bounds?.radius || 1;
    const project = (point, normalized = false) => {
      const [x, y, z] = normalized ? point : point.map((value, axis) => (value - center[axis]) / radius);
      const a = x * Math.cos(this.yaw) - y * Math.sin(this.yaw);
      const b = x * Math.sin(this.yaw) + y * Math.cos(this.yaw);
      const vertical = b * Math.sin(this.pitch) + z * Math.cos(this.pitch);
      return [width / 2 + a * scale, height / 2 + 15 - vertical * scale, b * Math.cos(this.pitch) - z * Math.sin(this.pitch)];
    };
    const line = (a, b, color) => {
      const p = project(a, true), q = project(b, true); ctx.strokeStyle = color;
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
    };
    for (let i = 0; i <= 4; i++) {
      const t = -1 + i * .5;
      line([-1, t, -1], [1, t, -1], '#e3ebf0');
      line([t, -1, -1], [t, 1, -1], '#e3ebf0');
      line([-1, 1, t], [1, 1, t], '#edf2f5');
    }
    ctx.font = '11px sans-serif'; ctx.textBaseline = 'middle';
    for (let axis = 0; axis < 3; axis++) {
      const start = [-1, -1, -1], end = [-1, -1, -1]; end[axis] = 1;
      line(start, end, '#7e91a3');
      const label = project(end, true); ctx.fillStyle = '#536176'; ctx.fillText(`SVD ${axis + 1}`, label[0] + 6, label[1] - 10);
      for (let i = 0; i <= 4; i++) {
        const t = -1 + i * .5, location = [...start]; location[axis] = t;
        const p = project(location, true);
        ctx.fillText((center[axis] + t * radius).toFixed(2), p[0] + 4, p[1] + 9);
      }
    }
    this.screenPoints = this.points.map((point) => ({ ...point, screen: project(point.value) })).sort((a, b) => a.screen[2] - b.screen[2]);
    this.screenCenters = (this.options.centers || []).map((point) => ({ ...point, screen: project(point.value) }));
    const selected = this.screenPoints.find((point) => point.id === this.options.selectedId);
    const assignedCenter = this.screenCenters.find((point) => point.cluster === selected?.cluster);
    if (selected && assignedCenter) {
      ctx.strokeStyle = '#087f8c'; ctx.lineWidth = 2.5; ctx.setLineDash([]); ctx.beginPath();
      ctx.moveTo(selected.screen[0], selected.screen[1]); ctx.lineTo(assignedCenter.screen[0], assignedCenter.screen[1]); ctx.stroke(); ctx.lineWidth = 1;
    }
    for (const point of this.screenPoints) {
      const isSelected = point.id === this.options.selectedId;
      ctx.globalAlpha = this.options.selectedCluster != null && point.cluster !== this.options.selectedCluster ? .23 : .9; ctx.fillStyle = point.color;
      ctx.beginPath(); ctx.arc(point.screen[0], point.screen[1], isSelected ? 9 : 3, 0, Math.PI * 2); ctx.fill();
      if (isSelected) { ctx.globalAlpha = 1; ctx.strokeStyle = '#173b52'; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1; }
    }
    ctx.globalAlpha = 1;
    for (const point of this.screenCenters) {
      const [x, y] = point.screen;
      const radius = point.cluster === this.options.selectedCluster ? 14 : 10;
      ctx.fillStyle = point.color; ctx.strokeStyle = '#173b52'; ctx.lineWidth = point.cluster === this.options.selectedCenter ? 3 : 1;
      ctx.beginPath(); ctx.moveTo(x, y - radius); ctx.lineTo(x + radius, y); ctx.lineTo(x, y + radius); ctx.lineTo(x - radius, y); ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.lineWidth = 1; ctx.fillStyle = '#173b52'; ctx.font = 'bold 12px sans-serif'; ctx.fillText(`C${point.cluster}${point.cluster === this.options.selectedCluster ? ' · assigned' : ''}`, x + radius + 3, y - radius);
    }
  }

  setBounds(points) {
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (const point of points) point.value.forEach((value, axis) => { min[axis] = Math.min(min[axis], value); max[axis] = Math.max(max[axis], value); });
    this.bounds = { center: min.map((value, axis) => (value + max[axis]) / 2), radius: Math.max(...max.map((value, axis) => (value - min[axis]) / 2), 1e-8) };
  }

  pick(event) {
    const box = this.canvas.getBoundingClientRect(), x = event.clientX - box.left, y = event.clientY - box.top;
    const center = (this.screenCenters || []).find((item) => Math.abs(x-item.screen[0])+Math.abs(y-item.screen[1]) <= (item.cluster===this.options.selectedCluster?14:10));
    return center || (this.screenPoints || []).slice().reverse().find((item) => Math.hypot(x-item.screen[0],y-item.screen[1]) < (item.id===this.options.selectedId?10:8));
  }

  hover(event) {
    const box = this.canvas.getBoundingClientRect(), x = event.clientX - box.left, y = event.clientY - box.top;
    const point = this.pick(event);
    this.tip.hidden = !point;
    if (!point) return;
    this.tip.innerHTML = this.tooltipHTML(point);
    this.tip.style.width = `${Math.max(80, Math.min(340, box.width - 16))}px`;
    const [left, top] = boundedTooltipPosition([x, y], [this.tip.offsetWidth, this.tip.offsetHeight], [box.width, box.height]);
    this.tip.style.left = `${left}px`; this.tip.style.top = `${top}px`;
  }

  dispose() { this.abort.abort(); this.observer.disconnect(); this.element.replaceChildren(); }
}
