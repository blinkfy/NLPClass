// Mirrors nlplab.information_extraction.rules; offsets always count Unicode code points.
const founding=['创立','创建','创办'],negation=['未','没有','并未','不曾','否认','未曾','不'];
const rules=[['创立','PER','ORG',founding],['任职于','PER','ORG',['任职于','就职于','供职于','就业于','加入']],['毕业于','PER','ORG',['毕业于']],['就读于','PER','ORG',['就读于']],['出生于','PER','LOC',['出生于','生于']],['位于','ORG','LOC',['总部位于','总部设在','坐落于','位于']]];
export function extractRelations(text,entities) {
  const chars=Array.from(text),slice=(a,b)=>chars.slice(a,b).join('');
  const find=(word,a,b)=>{const value=slice(a,b),at=value.indexOf(word);return at<0?-1:a+Array.from(value.slice(0,at)).length;};
  const ordered=[...entities].sort((a,b)=>a.start-b.start||a.end-b.end||a.type.localeCompare(b.type));
  for(let i=0;i<ordered.length;i++){const e=ordered[i];if(slice(e.start,e.end)!==e.text||e.start<0||e.end>chars.length||e.end<=e.start||(i&&ordered[i-1].end>e.start))throw new Error('NER 实体跨度无效或重叠');}
  function prefix(subject,start,name,between) {
    let cursor=subject.end;
    for(const entity of between.filter(e=>e.end<=start)) {
      const connector=slice(cursor,entity.start).trim();
      if((entity.type===subject.type&&['和','、'].includes(connector))||(name==='创立'&&entity.type==='LOC'&&connector==='在'))cursor=entity.end;
      else return false;
    }
    const trailing=slice(cursor,start).trim();
    return name==='创立'?['','曾','曾经'].includes(trailing):name==='位于'?['','的','总部','的总部'].includes(trailing):trailing==='';
  }
  function modifier(start,end,between) {
    if(end-start>22||!between.length||between[0].type!=='PER')return false;
    const founder=between[0],location=between[1];
    if(slice(start,founder.start).trim()||between.length>2||(location&&location.type!=='LOC'))return false;
    return founding.some(word=>{
      const at=find(word,founder.end,end);if(at<0)return false;
      const valid=location?slice(founder.end,location.start).trim()==='在'&&!slice(location.end,at).trim():['','曾','曾经'].includes(slice(founder.end,at).trim());
      return valid&&!negation.some(n=>slice(founder.end,at).includes(n))&&slice(at+Array.from(word).length,end).trim()==='的';
    });
  }
  const relations=[],seen=new Set();
  ordered.forEach((subject,i)=>{
    for(const object of ordered.slice(i+1))for(const [name,subjectType,objectType,triggers]of rules) {
      if(subject.type!==subjectType||object.type!==objectType)continue;
      const gap=slice(subject.end,object.start);
      if(object.start-subject.end>30||Array.from(gap).some(c=>'。！？!?；;\n\r，,'.includes(c)))continue;
      const between=ordered.filter(e=>subject.end<=e.start&&e.end<=object.start&&e!==subject&&e!==object),matches=[];
      for(const word of triggers) {
        const start=find(word,subject.end,object.start),end=start+Array.from(word).length;
        if(start<0||between.some(e=>e.start<end&&e.end>start)||negation.some(n=>slice(subject.end,start).includes(n))||!prefix(subject,start,name,between))continue;
        const tail=slice(end,object.start),after=between.filter(e=>e.start>=end);
        const repeated=name==='创立'&&triggers.some(other=>tail===`并${other}`);
        const simple=!after.length&&(repeated||(object.start-end<=5&&!Array.from(tail).some(c=>'说称表示认为访问参观创立创建创办'.includes(c))));
        if(!simple&&!(name==='任职于'&&modifier(end,object.start,after)))continue;
        matches.push({text:word,start,end});
      }
      matches.sort((a,b)=>a.start-b.start||Array.from(b.text).length-Array.from(a.text).length||a.text.localeCompare(b.text));
      if(!matches.length)continue;
      const key=[subject.start,subject.end,name,object.start,object.end].join('|');if(seen.has(key))continue;seen.add(key);
      relations.push({subject:{...subject},predicate:name,object:{...object},evidence:{text:slice(subject.start,object.end),start:subject.start,end:object.end},trigger:matches[0],rule_id:`${subjectType}-${name}-${objectType}`});
    }
  });
  return relations;
}
