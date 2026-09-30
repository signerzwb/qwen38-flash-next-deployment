#!/usr/bin/env python3
"""Verify an existing Strata Flash-Next endpoint; never install or start a model."""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request


def request(endpoint, key, body=None, timeout=30):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(endpoint, data=payload, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(endpoint, key, body):
    payload = dict(body, stream=True, stream_options={"include_usage": True})
    start = time.perf_counter()
    ttft = None
    text = []
    usage = {}
    done = False
    finish = None
    with request(endpoint + "/v1/chat/completions", key, payload, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise ValueError("The endpoint returned a streaming error")
            choices = event.get("choices") or []
            if choices:
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    text.append(content)
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
            if event.get("usage"):
                usage = event["usage"]
    if not done:
        raise ValueError("The stream ended without [DONE]")
    return {"text": "".join(text), "ttft_s": ttft, "total_s": time.perf_counter() - start,
            "finish_reason": finish, "usage": usage}


def long_document(word_count):
    rng = random.Random(20261001)
    vocabulary = ["audit", "register", "vendor", "migration", "staffing", "quarterly", "budget", "minutes", "review", "archived"]
    count = max(3, (word_count + 19) // 20)
    lines = ["[%05d] %s" % (i, " ".join(rng.choice(vocabulary) for _ in range(20))) for i in range(count)]
    lines[count // 2] = "KR-4471 的最终结算金额是 837462.19 元。"
    return "\n".join(lines) + "\n问题：KR-4471 的最终结算金额是多少元？只回答数字。"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Base address without /v1, e.g. http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-context", type=int)
    parser.add_argument("--key", default=os.environ.get("STRATA_API_KEY"), help="Prefer STRATA_API_KEY environment variable")
    parser.add_argument("--long-words", type=int, default=0, help="Optional synthetic long document size in words, not tokens")
    args = parser.parse_args()
    if not args.key or args.key == "REPLACE_WITH_YOUR_API_KEY":
        parser.error("Set your own STRATA_API_KEY or --key; placeholders are not accepted")
    if args.long_words < 0:
        parser.error("--long-words must be nonnegative")
    endpoint = args.endpoint.rstrip("/")
    failures = []

    def check(label, passed):
        print("%s %s" % ("PASS" if passed else "FAIL", label), flush=True)
        if not passed:
            failures.append(label)

    with request(endpoint + "/v1/models", args.key) as response:
        models = json.load(response).get("data") or []
    model = next((entry for entry in models if entry.get("id") == args.model), None)
    check("model ID", model is not None)
    context = ((model or {}).get("meta") or {}).get("n_ctx") or (model or {}).get("max_model_len")
    print("context=%s" % context)
    if args.expected_context is not None:
        check("context length", context is not None and int(context) == args.expected_context)
    try:
        with request(endpoint + "/v1/models", None):
            check("unauthenticated request rejected", False)
    except urllib.error.HTTPError as error:
        check("unauthenticated request rejected", error.code == 401)
    if failures:
        return 1

    body = {"model": args.model, "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
            "max_tokens": 200, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    short = stream_chat(endpoint, args.key, body)
    check("short conversation", bool(short["text"].strip()) and short["finish_reason"] == "stop")
    print(json.dumps({k: v for k, v in short.items() if k != "text"}, ensure_ascii=False))
    if args.long_words:
        body["messages"] = [{"role": "user", "content": long_document(args.long_words)}]
        body["max_tokens"] = 24
        for label in ("first long request (cache state unverified)", "repeated long request"):
            result = stream_chat(endpoint, args.key, body)
            check(label, "837462.19" in result["text"] and result["finish_reason"] == "stop")
            print(json.dumps({k: v for k, v in result.items() if k != "text"}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        # Do not dump requests, headers, or exception strings that could contain credentials.
        print("FAIL request/response error (%s)" % type(error).__name__, file=sys.stderr)
        sys.exit(1)
