from __future__ import print_function
import torch

x = torch.empty(5,3)
print(x)


x = torch.randn_like(x, dtype=torch.float)
print(x)
#torch는 튜플과 같으며 모든 튜플 연산에 사용할 수 있습니다.

print(x.size)

y = torch.rand(5,3)
print(x+y)

print(torch.add(x,y))

result = torch.empty(5,3)
torch.add(x, y, out= result)
print(result)

y.add_(x)
print(y)

print(x[:,1])

#view: 텐서의 크기나 모양을 변경하고 싶을 때 torch.view를 사용
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
#tensor에 하나의 값만 존재한다면, item을 이용해서 숫자 값을 얻을 수 있다
x1 = torch.randn(1)
print(x1)
print(x1.item())

#Torch tensor를 numpy 배열로 변환하거나, 그 반대로 하는 것은 매우 쉽다.
#Torch tensor와 numpy 배열은 저장 공간을 공유하기 때문에 하나를 변경하면 다른 하나도 변경됩니다.


a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# 넘파이 배열을 토치 텐서로 변환하기
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print("x",x)
    print("z",z)
    print(z.to("cpu", torch.double))


