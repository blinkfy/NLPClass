const sidebar=document.getElementById('site-sidebar'), toggle=document.getElementById('drawer-toggle'), backdrop=document.getElementById('drawer-backdrop');
const mobile=matchMedia('(max-width: 768px)');
function drawer(open) {
  document.body.classList.toggle('drawer-open',open);
  toggle.setAttribute('aria-expanded',String(open)); backdrop.hidden=!open;
  sidebar.inert=mobile.matches&&!open;
  if(open) (sidebar.querySelector('a[aria-current=page]')||sidebar.querySelector('a')).focus();
}
toggle.addEventListener('click',()=>drawer(toggle.getAttribute('aria-expanded')!=='true'));
backdrop.addEventListener('click',()=>{drawer(false);toggle.focus();});
mobile.addEventListener('change',()=>drawer(false));drawer(false);
document.addEventListener('keydown',event=>{
  if(!document.body.classList.contains('drawer-open')) return;
  if(event.key==='Escape') {drawer(false);toggle.focus();}
  if(event.key==='Tab') {
    const links=[...sidebar.querySelectorAll('a')],first=links[0],last=links.at(-1);
    if(event.shiftKey&&document.activeElement===first) {event.preventDefault();last.focus();}
    else if(!event.shiftKey&&document.activeElement===last) {event.preventDefault();first.focus();}
  }
});
const tabs=[...document.querySelectorAll('[data-tab]')];
export function activeTab() {return tabs.find(t=>t.getAttribute('aria-selected')==='true')?.dataset.tab;}
export function selectTab(key,updateHash=true) {
  if(!tabs.some(t=>t.dataset.tab===key)) return;
  for(const tab of tabs) {const active=tab.dataset.tab===key;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;document.getElementById(tab.dataset.tab).hidden=!active;}
  if(updateHash) history.replaceState(null,'',`#${key}`);
  document.dispatchEvent(new CustomEvent('module-tab',{detail:key}));window.dispatchEvent(new Event('resize'));
}
tabs.forEach((tab,i)=>{
  tab.addEventListener('click',()=>selectTab(tab.dataset.tab));
  tab.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight'].includes(e.key))return;e.preventDefault();const next=tabs[(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length];next.focus();selectTab(next.dataset.tab);});
});
function hashTab() {const raw=location.hash.slice(1),key=({metrics:'evaluation',parameters:'process'})[raw]||raw;selectTab(key||((window.NLPLAB_PAGE || location.pathname)==='/sequence-labeling'?'overview':(window.NLPLAB_PAGE || location.pathname)==='/tagging'?'pos':tabs[0]?.dataset.tab),false);}
window.addEventListener('hashchange',hashTab);hashTab();
