import { api } from './api.js';
const target=document.getElementById('public-data-status');
if(target) try{const value=(await api('/api/v1/datasets/public/status')).THUCNews;target.textContent=value.status==='ready'?`已准备 ${value.count} 篇全文，五类各120。固定 split：${Object.entries(value.split_counts).map(([name,n])=>`${name} ${n}`).join(' / ')}。模型待开发。`:'waiting for dataset：需要官方 THUCNews 全文数据。';}catch(error){target.textContent=error.message;}
