

def recursion_function(self, left, right, accuracy, iterations_tpm, medi):

    if right - left < accuracy:
        return (right + left) / 2, iterations_tpm

    x_1 = (right - left)/4 + left
    x_2 = (right - left)/4 * 2 + left

    f_1 = self.target_function(x_1)
    f_2 = medi

    iterations_tpm += 1

    if f_1 <= f_2:
        return recursion_function(self, left, x_2, accuracy, iterations_tpm, f_1)
    else:
        x_3 = (right - left) / 4 * 3 + left
        f_3 = self.target_function(x_3)
        iterations_tpm += 1
        if f_2 <= f_3:
            return recursion_function(self, x_1, x_3, accuracy, iterations_tpm, f_2)
        else:
            return recursion_function(self, x_2, right, accuracy, iterations_tpm, f_3)


def trial_point_method(self, accuracy):
    x_2 = (self.right_border - self.left_border) / 4 * 2 + self.left_border
    f_2 = self.target_function(x_2)
    return recursion_function(self, self.left_border, self.right_border, accuracy, 1, f_2)


