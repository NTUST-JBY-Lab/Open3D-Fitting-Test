from open3d_fitting_test.Util import rotFromToVec, rotAlignUpFwd
import numpy as np

def testRotFromToVec():
    vec1 = np.array([0, 1, 0])
    vec2 = np.array([1, 0, 0])

    rot = rotFromToVec(vec1, vec2)

    assert np.all(np.isclose(rot @ vec1, vec2))
    assert np.all(np.isclose(np.linalg.inv(rot) @ vec2, vec1))

def testRotAlignUpFwd():
    up = np.array([1, 0, 0])
    fwd = np.array([0, 1, 0])
    right = np.cross(fwd, up)

    rot = rotAlignUpFwd(up, fwd)

    assert np.all(np.isclose(rot @ fwd, np.array([0, 0, 1])))
    assert np.all(np.isclose(rot @ up, np.array([0, 1, 0])))
    assert np.all(np.isclose(rot @ right, np.array([1, 0, 0])))

    rot = np.linalg.inv(rot)

    assert np.all(np.isclose(rot @ np.array([0, 0, 1]), fwd))
    assert np.all(np.isclose(rot @ np.array([0, 1, 0]), up))
    assert np.all(np.isclose(rot @ np.array([1, 0, 0]), right))

if __name__ == "__main__":
    testRotFromToVec()
    testRotAlignUpFwd()
    print("All pass")
