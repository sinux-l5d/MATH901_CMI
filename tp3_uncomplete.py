import regex as re
import numpy as np
np.seterr(divide='ignore')
np.set_printoptions(precision=3, suppress=True)


mat = [[3, 2, 1, 0, 0, 1800],
       [1, 0, 0, 1, 0, 400],
       [0, 1, 0, 0, 1, 600],
       [30, 50, 0, 0, 0, 0]]
mat = np.float64(mat)


def getK(mat):
    feco = mat[-1, :-1]  # don't take the bottom right value
    return feco.argmax()


def getL(mat):
    secmembre = mat[:-1, -1]
    k = getK(mat)
    coef = mat[:, k]
    out = np.zeros(len(secmembre))

    for i in range(len(secmembre)):
        try:
            out[i] = secmembre[i] / coef[i]
        except ZeroDivisionError:
            out[i] = float("inf")
    return out.argmin()


def simplex(mat):
    while mat[-1, :].max() > 0:
        k = getK(mat)
        print(f"K = {k+1}")
        print(f"coef K = {mat[-1, k]}")
        l = getL(mat)
        print(f"L = {l+1}")
        print(f"coef L = {mat[l, -1]}")
        pivot = mat[l, k]
        print(f"pivot = {pivot}")

        mat[l, :] /= pivot

        for i in range(len(mat)):
            if i != l:
                mat[i, :] -= (mat[i, k] * mat[l, :]) / pivot


reSingleVar = r"(?P<sign>[+\-]?)\s*(?P<weight>\d*)(?P<var>[a-zA-Z])"
reWeightGroup = rf"(?P<weights>{reSingleVar}\s*)+"
reFuncEco = rf"(?P<func>max) (?P<funcparam>[a-zA-Z])\s*=\s*({reWeightGroup})"
reOperatorInequality = r"(?P<operator>[<>]=)"
reSecMemb = r"(?P<secmember>\d+)"
reInequality = rf"{reWeightGroup}\s*{reOperatorInequality}\s*{reSecMemb}"


def parse(instructions: list[str]) -> np.ndarray:
    """Parse a system of inequalities into a matrix
        It must have exactly one max instruction with as many variables as you want. Example:
            max z = 30x + 50y (+ 40u + 60v...)
        and then as many inequalities as you want. Example:
            3x + 2y (+ 4u + 5v...) <= 1800
            x <= 400
            y <= 600
    """
    instructions = [ins.strip() for ins in instructions]

    # Fonction économique
    economic_function = [ins for ins in instructions if ins.startswith("max")]
    assert len(economic_function) == 1, "You must have exactly one max instruction"

    economic_function = economic_function[0]
    m = re.match(reFuncEco, economic_function)

    # remove the function from the list of instructions
    instructions.remove(economic_function)

    vars = []
    for part in m.captures("weights"):
        p = re.match(reSingleVar, part)
        assert p is not None, f"Invalid variable: {part}"
        w = int(p.group("sign") + p.group("weight"))
        v = p.group("var")
        vars.append((w, v))

    vars.sort(key=lambda x: x[1])
    assert len(vars) == len(set([v for _, v in vars])
                            ), "Cannot have two variables with the same name"
    print(vars)

    # Inégalités
    inequalities = []
    for inequality in instructions:
        m = re.match(reInequality, inequality)
        assert m is not None, f"Invalid inequality: {inequality}"
        ineq = {"op": m.captures("operator"), "secmember": m.captures(
            "secmember"), "vars": []}
        for part in m.captures("weights"):
            p = re.match(reSingleVar, part)
            assert p is not None, f"Invalid variable: {part}"
            print(p.groupdict())
            w = p.group("weight")
            if w == "":
                w = "1"
            w = int(p.group("sign") + w)
            v = p.group("var")
            ineq["vars"].append((w, v))
        ineq["vars"].sort(key=lambda x: x[1])
        inequalities.append(ineq)

    print(inequalities)


if __name__ == "__main__":
    # simplex(mat)
    # print(mat)
    print(reFuncEco)
    m = re.match(reFuncEco, "max z = 30x + 50y + 40u + 60v")
    print("func", m.captures("func"))
    print("funcparam", m.captures("funcparam"))
    print("weight", m.captures("weights"))
    parse(["max z = 30x + 50y", "3x + 2y <= 1800", "x <= 400", "y <= 600"])
