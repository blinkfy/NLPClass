/* Contiguous original-character context, not a new tokenizer. */
export function sentenceContext(text,span,maximum) {
  const chars=Array.from(text||''),target=span?.start??0;
  let start=target,end=span?.end??Math.min(1,chars.length);
  while(start>0&&!/[。！？!?\n]/.test(chars[start-1]))start--;
  while(end<chars.length&&!/[。！？!?\n]/.test(chars[end-1]))end++;
  if(end-start>maximum&&span&&span.end-span.start<=maximum){start=span.start;end=span.end;}
  return {start,end,text:chars.slice(start,end).join(''),oversize:end-start>maximum};
}

export function splitDocuments(text) {
  const documents=[],separator=/^[ \t]*---[ \t]*(?:\r?\n|$)/gm;let start=0;
  for(const match of text.matchAll(separator)){let end=match.index;
    if(end>start&&text.slice(end-2,end)==='\r\n')end-=2;else if(end>start&&text[end-1]==='\n')end--;
    documents.push(text.slice(start,end));start=match.index+match[0].length;
  }documents.push(text.slice(start));return documents;
}
