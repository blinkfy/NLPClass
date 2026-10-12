/* Preserve the selected control through coordinated-view redraws. */
export function retainViewFocus(root, render) {
  const active = document.activeElement;
  const hostId = root?.contains(active) ? active.closest('[id]')?.id : null;
  const attribute = root?.contains(active) ? [...active.attributes].find(a => /^data-(cm|bert|bilstm|llm|boundary)-/.test(a.name)) : null;
  const disclosures = [...(root?.querySelectorAll('details[open]') || [])].map(node => ({node, hostId:node.closest('[id]')?.id, title:node.querySelector(':scope > summary')?.textContent}));
  const result = render();
  for (const saved of disclosures) {
    if (saved.node.isConnected || !saved.title) continue;
    const host = document.getElementById(saved.hostId) || root;
    const replacement = [...(host?.querySelectorAll('details') || [])].find(node => node.querySelector(':scope > summary')?.textContent === saved.title);
    if (replacement) replacement.open = true;
  }
  if (attribute && !active.isConnected) {
    const host = document.getElementById(hostId) || root;
    host?.querySelector(`[${attribute.name}="${CSS.escape(attribute.value)}"]`)?.focus({preventScroll:true});
  }
  return result;
}
