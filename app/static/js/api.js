export async function api(url, options = {}) {
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) {
    const message =
      value.error?.message ||
      (Array.isArray(value.detail)
        ? value.detail.map((x) => x.msg).join("；")
        : "请求失败");
    throw new Error(message);
  }
  return value;
}
export function node(tag, text = "", className = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  element.className = className;
  return element;
}
export function notice(element, message = "", failure = false) {
  element.hidden = !message;
  element.textContent = message;
  element.className = `notice${failure ? " failure" : ""}`;
}
