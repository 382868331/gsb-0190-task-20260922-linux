"""输入校验与错误语义测试。"""

import math
import unittest

from iso3d import InvalidInputError, ScalarField3D, extract_isosurface


def flat(ni, nj, nk, func):
    return tuple(
        float(func(i, j, k))
        for k in range(nk)
        for j in range(nj)
        for i in range(ni)
    )


class FieldValidationTest(unittest.TestCase):
    def test_valid_construction(self):
        field = ScalarField3D(ni=2, nj=3, nk=4, values=flat(2, 3, 4, lambda i, j, k: i + j + k))
        self.assertEqual(field.value_at(1, 2, 3), 6.0)
        self.assertEqual(field.node_id(1, 2, 3), (3 * 3 + 2) * 2 + 1)

    def test_bool_rejected_for_dims(self):
        with self.assertRaisesRegex(InvalidInputError, "ni"):
            ScalarField3D(ni=True, nj=2, nk=2, values=(0.0,) * 4)

    def test_non_int_rejected_for_dims(self):
        with self.assertRaisesRegex(InvalidInputError, "nj"):
            ScalarField3D(ni=2, nj=2.5, nk=2, values=(0.0,) * 4)

    def test_zero_dim_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "nk"):
            ScalarField3D(ni=2, nj=2, nk=0, values=())

    def test_dim_above_limit_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "ni"):
            ScalarField3D(ni=17, nj=1, nk=1, values=(0.0,) * 17)

    def test_max_grid_accepted(self):
        field = ScalarField3D(ni=16, nj=16, nk=16, values=(0.0,) * 4096)
        self.assertEqual(field.value_at(15, 15, 15), 0.0)

    def test_values_length_mismatch(self):
        with self.assertRaisesRegex(InvalidInputError, "values"):
            ScalarField3D(ni=2, nj=2, nk=2, values=(0.0,) * 7)

    def test_values_not_sequence(self):
        with self.assertRaisesRegex(InvalidInputError, "values"):
            ScalarField3D(ni=2, nj=2, nk=2, values="01234567")

    def test_nan_rejected_with_location(self):
        values = [0.0] * 8
        values[5] = math.nan
        with self.assertRaisesRegex(InvalidInputError, r"values\[5\]"):
            ScalarField3D(ni=2, nj=2, nk=2, values=values)

    def test_infinity_rejected_with_location(self):
        values = [0.0] * 8
        values[3] = math.inf
        with self.assertRaisesRegex(InvalidInputError, r"values\[3\]"):
            ScalarField3D(ni=2, nj=2, nk=2, values=values)
        values[3] = -math.inf
        with self.assertRaisesRegex(InvalidInputError, r"values\[3\]"):
            ScalarField3D(ni=2, nj=2, nk=2, values=values)

    def test_bool_value_rejected(self):
        values = [0.0] * 8
        values[0] = True
        with self.assertRaisesRegex(InvalidInputError, r"values\[0\]"):
            ScalarField3D(ni=2, nj=2, nk=2, values=values)

    def test_int_values_promoted_to_float(self):
        field = ScalarField3D(ni=2, nj=1, nk=1, values=(0, 3))
        self.assertEqual(field.values, (0.0, 3.0))
        self.assertIsInstance(field.values[0], float)


class IsoValidationTest(unittest.TestCase):
    def setUp(self):
        self.field = ScalarField3D(ni=2, nj=2, nk=2, values=(0.0,) * 8)

    def test_nan_iso_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "iso"):
            extract_isosurface(self.field, math.nan)

    def test_inf_iso_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "iso"):
            extract_isosurface(self.field, math.inf)

    def test_bool_iso_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "iso"):
            extract_isosurface(self.field, True)

    def test_string_iso_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "iso"):
            extract_isosurface(self.field, "1.0")

    def test_non_field_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "field"):
            extract_isosurface(((0.0,) * 8,), 1.0)

    def test_failed_call_leaves_field_untouched(self):
        before = self.field.values
        with self.assertRaises(InvalidInputError):
            extract_isosurface(self.field, math.nan)
        self.assertEqual(self.field.values, before)


class NestedConstructionTest(unittest.TestCase):
    def test_roundtrip(self):
        nested = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
        field = ScalarField3D.from_nested(nested)
        self.assertEqual((field.ni, field.nj, field.nk), (2, 2, 2))
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    self.assertEqual(field.value_at(i, j, k), nested[k][j][i])

    def test_ragged_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "nested"):
            ScalarField3D.from_nested([[[1.0, 2.0]], [[1.0]]])

    def test_empty_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "nested"):
            ScalarField3D.from_nested([])


if __name__ == "__main__":
    unittest.main()
