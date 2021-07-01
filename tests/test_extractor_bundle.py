from unittest import TestCase
import numpy as np
import orcasong.extractors.bundles as bundles


class TestGetPlanePositions(TestCase):
    def setUp(self) -> None:
        self.positions = np.array([
            [0, 0, 0],
            [0, 0, 5],
            [5, 3, 2],
            [0, 0, 2],
        ])
        self.directions = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, -1],
        ])

    def test_positions_flat_plane(self):
        plane_point = np.array([0, 0, 0])
        plane_normal = np.array([0, 0, 1])
        result = bundles.get_plane_positions(
            self.positions, self.directions, plane_point, plane_normal,
        )
        target = np.array([
            [0, 0],
            [0, 0],
            [5, 3],
            [0, 2],
        ])
        np.testing.assert_array_equal(result, target)

    def test_positions_flat_plane_no_plane_normal(self):
        plane_point = np.array([0, 0, 0])
        with self.assertRaises(ValueError):
            bundles.get_plane_positions(
                self.positions, self.directions, plane_point,
            )

    def test_positions_shifted_plane(self):
        plane_point = np.array([1, 0, 0])
        plane_normal = np.array([0, 0, 1])
        result = bundles.get_plane_positions(
            self.positions, self.directions, plane_point, plane_normal,
        )
        target = np.array([
            [-1, 0],
            [-1, 0],
            [4, 3],
            [-1, 2],
        ])
        np.testing.assert_array_equal(result, target)

    def test_positions_angled_plane(self):
        plane_point = np.array([0, 0, 0])
        plane_normal = np.array([1, 0, 1])
        result = bundles.get_plane_positions(
            self.positions, self.directions, plane_point, plane_normal,
        )
        target = np.array([
            [0, 0],
            [0, 0],
            [np.sqrt(50), 3],
            [0, 2],
        ])
        np.testing.assert_array_equal(result, target)

    def test_positions_angled_plane_2(self):
        result = bundles.get_plane_positions(
            positions=np.array([[-1, 0, 1]]),
            directions=np.array([[1, 0, -1]]),
            plane_point=np.array([0, 0, 0]),
            plane_normal=np.array([-1, 0, 1]),
        )
        np.testing.assert_array_equal(result, np.array([[0, 0]]))


class TestPairwiseDistances(TestCase):
    def setUp(self) -> None:
        self.positions_plane = np.array([
            [0, -3],
            [4, 0],
            [0, 3],
        ])

    def test_matrix(self):
        result = bundles.get_pairwise_distances(
            positions_plane=self.positions_plane,
            as_matrix=True,
        )
        target = np.array([
            [0, 5, 6],
            [5, 0, 5],
            [6, 5, 0],
        ])
        np.testing.assert_array_equal(result, target)

    def test_flat(self):
        result = bundles.get_pairwise_distances(
            positions_plane=self.positions_plane,
        )
        target = np.array([5, 6, 5])
        np.testing.assert_array_equal(result, target)
