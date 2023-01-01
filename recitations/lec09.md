## Khởi tạo trọng số https://youtu.be/CukpVt-1PA4?t=4194
Ôn lại bài trước, ta có công thức cập nhật trọng số bằng SGD
`W_i := W_i - alpha gradient_{W_i} l(h_theta(X),y)`

Vấn đề ở đây là với i = 1, ta khởi tạo trọng số của W_i, b_i như thế nào? set toàn bộ = 0?

Ôn lại công thức backprob mà không dùng bias:
- `Z_i+1 = sigma_i(W_i Z_i)`
- `G_i = (G_i+1 sigma_i'(Z_i W_i)) W_iT^`

Nếu W_i = 0 thì G_j = 0 for j <= i => gradient_{W_i} l(h_theta(X),y) = 0 => tham số không được cập nhật.
Vậy nên khởi tạo W_i = 0 là một lựa chọn vô cùng tồi.

![](files/lec06-12.png)



## Chuẩn hóa https://youtu.be/ky7qiKyZmnE?t=742
...