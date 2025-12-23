import importlib


class Factory:
    @staticmethod
    def create(step_config: dict):
        """
        step_config:
          module: "features.sequence"
          class: "SequenceEncoder"
          params: {...}
        """
        try:
            module_path = step_config["module"]
            class_name = step_config["class"]
            params = step_config.get("params", {})

            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            return cls(**params)

        except Exception as e:
            raise RuntimeError(
                f"Factory failed to create component "
                f"(module={step_config.get('module')}, "
                f"class={step_config.get('class')})"
            ) from e
