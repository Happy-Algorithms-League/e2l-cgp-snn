import cgp


class CustomConstantFloatTen(cgp.ConstantFloat):
    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)
        self._output = 10.0


class CustomConstantFloatOne(cgp.ConstantFloat):
    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)
        self._output = 1.0


class CustomConstantFloatPointOne(cgp.ConstantFloat):
    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)
        self._output = 0.1
