from keras import Model


class BaseModel(Model):

    def filer_inputs(self, inputs: dict) -> dict:
        return {k: v for k, v in inputs.items() if k in self.inputs}

    def inspect_signatures(self, model: Model) -> dict:
        """Inspect the model signatures.

        Args:
            self: write your description
            model: write your description
        """
        sig_keys = list(model.signatures.keys())
        logger.info(f"found signatures: {sig_keys}")
        info = {}
        for sig in sig_keys:
            _infer = model.signatures[sig]
            _inputs = _infer.structured_input_signature
            _outputs = _infer.structured_outputs
            info["signature"] = {
                "inputs": _inputs,
                "outputs": _outputs,
            }
        return info