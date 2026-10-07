"""Writes pkg/chattemplate/testdata/oracle.json from the real Jinja template (via serve/frontend.py's ChatTemplate, the
repository's HF-equivalent renderer). The Go renderer must reproduce every case, errors included.
Usage: python tools/gen_chat_fixture.py <template.jinja> [out.json]"""
import itertools, json, pathlib, random, sys
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


class TemplateRequestError(jinja2.exceptions.TemplateError, ValueError):
    pass


class ChatTemplate:
    """The same Jinja settings as transformers' apply_chat_template (and as the retired serve/frontend.py)."""

    def __init__(self, path):
        def raise_exception(message):
            raise TemplateRequestError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string(pathlib.Path(path).read_text(encoding="utf-8"))

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        # completed empty assistant turns (not the last message) are not used as examples for the next reply
        messages = [m for i, m in enumerate(messages)
                    if i == len(messages) - 1 or not (m.get("role") == "assistant" and not (m.get("content") or "").strip()
                                                      and not m.get("tool_calls"))]
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)


tpl = ChatTemplate(sys.argv[1])

U = lambda t: {"role": "user", "content": t}
A = lambda t, **k: {"role": "assistant", "content": t, **k}
S = lambda t: {"role": "system", "content": t}
T = lambda t: {"role": "tool", "content": t}
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather ☃ <b>", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer", "default": 3}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "calc", "description": "x", "parameters": {"type": "object", "properties": {}}}}]
call = lambda name, args: {"type": "function", "function": {"name": name, "arguments": args}}
convs = [
    [U("Hello")], [S("Be brief."), U("Hi")], [S("a"), S("b"), U("Hi")], [U("Hi"), A("Hello!"), U("again")],
    [U("q"), A("answer", reasoning_content="because"), U("q2")], [U("q"), A("answer", reasoning_content="  because \n"), U("q2"), A("a2", reasoning_content="r2")],
    [U("q"), A(""), U("q2")], [U("q"), A("", tool_calls=[call("get_weather", {"city": "Paris"})]), T("sunny"), A("It is sunny.")],
    [U("q"), A("let me check", tool_calls=[call("get_weather", {"city": "Paris", "days": 2}), call("calc", {"expr": "1+1", "flag": True, "list": [1, 2, "x"], "obj": {"k": "vé"}})]), T("r1"), T("r2"), U("thanks")],
    [U("q"), A("", tool_calls=[call("calc", {})]), T("x")], [U("<tool_response>\nx\n</tool_response>")], [U("  padded \n"), A("  out  ")],
    [U("Unicode 你好 \U0001f600")], [U("")], [S("")], [S("   "), U("x")], [{"role": "developer", "content": "dev rules"}, U("x")],
    [U("a"), S("late system")], [U("a"), {"role": "narrator", "content": "x"}], [A("first is assistant"), U("x")], [],
    [T("orphan tool"), U("x")], [S("s"), T("tool right after system")],
    [U("a"), A("b", tool_calls=[call("f", {"code": "print('hi')\nx = {\"a\": 1}"})]), T("ok")],
    [U("a"), A("", tool_calls=[{"function": {"name": "", "arguments": {}}}])],
    [U("q"), A("c", tool_calls=[call("f", "")]), T("x")], [U("q"), A("c", tool_calls=[call("f", '{"a": 1}')]), T("x")],
    [U("q"), A("c", tool_calls=[call("f", 5)]), T("x")],
]
kw_sets = [{}, {"add_generation_prompt": False}, {"enable_thinking": False}, {"enable_thinking": True, "reasoning_effort": "low"},
           {"reasoning_effort": "medium"}, {"reasoning_effort": "high"}, {"reasoning_effort": "bogus"}, {"preserve_thinking": False},
           {"preserve_thinking": False, "enable_thinking": False}]
cases = []
for conv in convs:
    for tl in (None, tools):
        for kw in kw_sets:
            # keep the matrix manageable: full kw sweep only for tool-less short conversations plus a few with tools
            if tl and kw not in ({}, {"enable_thinking": False}, {"reasoning_effort": "low"}):
                continue
            case = {"messages": conv, "tools": tl, "kwargs": kw}
            try:
                case["rendered"] = tpl.render(conv, tools=tl, **kw); case["error"] = None
            except Exception as e:
                case["rendered"] = None; case["error"] = type(e).__name__
            cases.append(case)
pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "pkg/chattemplate/testdata/oracle.json").parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "pkg/chattemplate/testdata/oracle.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
print(len(cases), "cases,", sum(1 for c in cases if c["error"]), "errors")
