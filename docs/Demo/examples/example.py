"""Example module for literalinclude demo."""


def normalize(x):
    """Normalize a vector to unit length."""
    return x / (x ** 2).sum() ** 0.5


# start example
class ScaledDotProduct:
    """Scaled dot-product attention."""

    def __init__(self, d_k: int):
        self.scale = d_k ** -0.5

    def forward(self, q, k, v):
        scores = (q @ k.transpose(-2, -1)) * self.scale
        weights = scores.softmax(dim=-1)
        return weights @ v
# end example


if __name__ == "__main__":
    print("example")
