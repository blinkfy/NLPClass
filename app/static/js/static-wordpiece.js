// BERT/ELECTRA BasicTokenizer + greedy WordPiece, using the exported vocabulary.
const dictionaries=new WeakMap();
const chinese=c=>{const n=c.codePointAt(0);return [[0x4e00,0x9fff],[0x3400,0x4dbf],[0x20000,0x2a6df],[0x2a700,0x2b73f],[0x2b740,0x2b81f],[0x2b820,0x2ceaf],[0xf900,0xfaff],[0x2f800,0x2fa1f]].some(([a,b])=>a<=n&&n<=b);};
const punctuation=c=>{const n=c.codePointAt(0);return (n>=33&&n<=47)||(n>=58&&n<=64)||(n>=91&&n<=96)||(n>=123&&n<=126)||/\p{P}/u.test(c);};
export function wordpieceEncoding(text,vocabulary,{lowercase=true,stripAccents=lowercase}={}) {
  if(!dictionaries.has(vocabulary))dictionaries.set(vocabulary,new Map(vocabulary.map((word,i)=>[word,i])));
  const vocab=dictionaries.get(vocabulary),output=[],offsets=[];let base=0;
  for(const chunk of text.split(/(\[UNK\]|\[SEP\]|\[PAD\]|\[CLS\]|\[MASK\])/)) {
    if(!chunk)continue;
    const chunkStart=base;base+=Array.from(chunk).length;
    if(/^\[(UNK|SEP|PAD|CLS|MASK)\]$/.test(chunk)){output.push(vocab.get(chunk));offsets.push([chunkStart,base]);continue;}
    const sources=[];
    Array.from(chunk).forEach((c,i)=>{
      const n=c.codePointAt(0);if(n===0||n===0xfffd||(/\p{C}/u.test(c)&&!['\t','\n','\r'].includes(c)))return;
      let part=[' ','\t','\n','\r'].includes(c)||/\p{Z}/u.test(c)?' ':chinese(c)?` ${c} `:c;
      if(lowercase)part=part.toLowerCase();
      if(stripAccents)part=part.normalize('NFD').replace(/\p{Mn}/gu,'');
      Array.from(part).forEach(()=>sources.push(i+chunkStart));
    });
    let cleaned=Array.from(chunk).map(c=>{
      const n=c.codePointAt(0);
      if(n===0||n===0xfffd||(/\p{C}/u.test(c)&&!['\t','\n','\r'].includes(c)))return '';
      if([' ','\t','\n','\r'].includes(c)||/\p{Z}/u.test(c))return ' ';
      return chinese(c)?` ${c} `:c;
    }).join('');
    if(lowercase)cleaned=cleaned.toLowerCase();
    if(stripAccents)cleaned=cleaned.normalize('NFD').replace(/\p{Mn}/gu,'');
    const basics=[];let current='',places=[];
    Array.from(cleaned).forEach((c,i)=>{if(c===' '||punctuation(c)){if(current)basics.push([current,places]);current='';places=[];if(c!==' ')basics.push([c,[sources[i]]]);}else{current+=c;places.push(sources[i]);}});
    if(current)basics.push([current,places]);
    for(const [word,places] of basics) {
      const chars=Array.from(word),pieces=[];let start=0,unknown=chars.length>100;
      const spans=[];
      while(!unknown&&start<chars.length){let end=chars.length,id;
        for(;end>start;end--){const value=(start?'##':'')+chars.slice(start,end).join('');if(vocab.has(value)){id=vocab.get(value);break;}}
        if(id===undefined){unknown=true;break;}pieces.push(id);spans.push([places[start],places[end-1]+1]);start=end;
      }
      output.push(...(unknown?[vocab.get('[UNK]')]:pieces));
      offsets.push(...(unknown?[[places[0],places.at(-1)+1]]:spans));
    }
  }
  return {ids:output,offsets};
}
export function wordpieces(text,vocabulary,options){return wordpieceEncoding(text,vocabulary,options).ids;}
