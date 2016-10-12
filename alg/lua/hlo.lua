num = 42
s = 'walternate'
u =[[Double brackets start and end multi-line strings.]]
t = nil

while num <50 do
	num = num + 1
	--io.write(num..'\n')
	print(num)
end

if num > 40 then
	print('over 40')
elseif s ~= 'walternate' then
	io.write('not over 40\n')
else
	thisIsGlobal = 5
	local line = io.read()
	print('Winter is coming, '..line)
end

karlSum = 0
for i = 1,100 do
	karlSum = karlSum + i
end


fredSum = 0
for j =100,1,-1 do fredSum = fredSum + j 
	--io.write(fredSum..'\n')
end

repeat
	print('the way of the future')
	num = num -1
until num == 0

function fib(n)
	if n<2 then 
		return 1
	end
	return fib(n-1)+fib(n-2)
end

function adder(x)
	return function(y) return x+y end
	end

a1 = adder(9)
a2 = adder(36)
print(a1(36))
print(a2(64))
print(fib(5))



x,y,z =1,2,3,4
function bar(a,b,c)
	print(a,b,c)
	return 4,8,15,16,23,42
end

x,y =bar('zaphod')
print(x,y)
f = function(x) return x*x end


