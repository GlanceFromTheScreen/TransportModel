from math import sqrt


gold = (3-sqrt(5))/2


def golden_search(self, accuracy):
    eps = accuracy
    remember_left_board = self.left_border
    remember_right_board = self.right_border

    count = 0
    x1 = self.left_border + gold*(self.right_border - self.left_border)
    x2 = self.right_border - gold*(self.right_border - self.left_border)
    count+=1
    fx1 = self.target_function(x1)
    count+=1
    fx2 = self.target_function(x2)
    while (abs(self.right_border - self.left_border) > eps):
        if fx1 < fx2:
            self.right_border = x2
            x2 = x1
            fx2 = fx1
            x1 = self.left_border + gold *(self.right_border - self.left_border)
            count += 1
            fx1 = self.target_function(x1)
        else:
            self.left_border = x1
            x1 = x2
            fx1 = fx2
            x2 = self.right_border - gold*(self.right_border - self.left_border)
            count += 1
            fx2 = self.target_function(x2)
    res_dot = (self.left_border + self.right_border) / 2

    self.left_border = remember_left_board
    self.right_border = remember_right_board
    return res_dot, count



