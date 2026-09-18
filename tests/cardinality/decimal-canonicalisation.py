from decimal import Decimal, localcontext

from data_quality.cardinality.hashing import canonicalize_scalar


a = canonicalize_scalar(
    Decimal("1.2300")
)

b = canonicalize_scalar(
    Decimal("1.23")
)

print(a)
print(b)

assert a == b
assert a == b"d0:123:-2"

print("PASS: Decimal trailing-zero canonicalisation")