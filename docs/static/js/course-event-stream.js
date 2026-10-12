/* Incremental SSE decoding: HTTP chunks need not align with UTF-8 or events. */
export async function consumeEventStream(response, onEvent) {
  if (!response.body) throw new Error('浏览器不支持流式响应。');
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = '', terminal = false;
  function flush() {
    while (true) {
      const boundary = /\r?\n\r?\n/.exec(buffer);
      if (!boundary) return;
      const frame = buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary[0].length);
      let event = 'message';
      const data = [];
      for (const line of frame.split(/\r?\n/)) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
      }
      if (data.length) {
        onEvent(event, JSON.parse(data.join('\n')));
        if (event === 'done' || event === 'error') { terminal = true; return; }
      }
    }
  }
  try {
    while (!terminal) {
      const {value, done} = await reader.read();
      if (done) { buffer += decoder.decode(); flush(); break; }
      buffer += decoder.decode(value, {stream: true});
      flush();
    }
    if (!terminal) throw new Error('生成连接中断，尚未收到完整结果。');
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
