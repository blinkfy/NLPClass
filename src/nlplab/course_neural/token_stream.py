"""Generation callback: real cumulative decoding, no synthetic typing animation."""


class TokenStream:
    def __init__(self, tokenizer, emit):
        self.tokenizer, self.emit = tokenizer, emit
        self.prompt_pending = True
        self.ids = []

    def put(self, value):
        # generate() first sends the complete prompt; it is not output text.
        if self.prompt_pending:
            self.prompt_pending = False
            return
        values = value.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        self.ids.extend(values)
        # A byte-level token may end inside a Chinese UTF-8 character. Publish
        # the cumulative decode, avoiding a temporary trailing replacement char.
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True).rstrip("\ufffd")
        self.emit("token", {"response": text, "token_count": len(self.ids)})

    def end(self):
        pass
