
Ŕ%%
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
PyFunc
input2Tin
output2Tout"
tokenstring"
Tin
list(type)("
Tout
list(type)(

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
@
Softplus
features"T
activations"T"
Ttype:
2

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.14.02v1.14.0-rc1-22-gaf24dc9¨
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h*
shape:˙˙˙˙˙˙˙˙˙h
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_2Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_4Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
h
Placeholder_5Placeholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
Placeholder_6Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
N
Placeholder_7Placeholder*
dtype0*
_output_shapes
: *
shape: 
N
Placeholder_8Placeholder*
shape: *
dtype0*
_output_shapes
: 
Ľ
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"h      *"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
:

.pi/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *Ľ2ž*"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
: 

.pi/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ľ2>*"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
: 
ď
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	h*

seed *
T0*"
_class
loc:@pi/dense/kernel*
seed2
Ú
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@pi/dense/kernel
í
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	h*
T0*"
_class
loc:@pi/dense/kernel
ß
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	h
Š
pi/dense/kernel
VariableV2*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	h*
dtype0*
_output_shapes
:	h
Ô
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	h*
use_locking(*
T0

pi/dense/kernel/readIdentitypi/dense/kernel*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	h

pi/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    * 
_class
loc:@pi/dense/bias

pi/dense/bias
VariableV2*
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ż
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
u
pi/dense/bias/readIdentitypi/dense/bias* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0

pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
pi/dense/TanhTanhpi/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes
:

0pi/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *×łÝ˝*$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes
: 

0pi/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *×łÝ=*$
_class
loc:@pi/dense_1/kernel
ö
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*

seed *
T0*$
_class
loc:@pi/dense_1/kernel*
seed2*
dtype0* 
_output_shapes
:

â
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
: 
ö
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

č
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

Ż
pi/dense_1/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *$
_class
loc:@pi/dense_1/kernel
Ý
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

pi/dense_1/kernel/readIdentitypi/dense_1/kernel*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:


!pi/dense_1/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@pi/dense_1/bias*
dtype0*
_output_shapes	
:
Ą
pi/dense_1/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@pi/dense_1/bias
Ç
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
{
pi/dense_1/bias/readIdentitypi/dense_1/bias*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:

pi/dense_1/MatMulMatMulpi/dense/Tanhpi/dense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
^
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *$
_class
loc:@pi/dense_2/kernel

0pi/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *Ş7ž*$
_class
loc:@pi/dense_2/kernel

0pi/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ş7>*$
_class
loc:@pi/dense_2/kernel*
dtype0*
_output_shapes
: 
ő
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*$
_class
loc:@pi/dense_2/kernel*
seed2.
â
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@pi/dense_2/kernel
ő
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
ç
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel
­
pi/dense_2/kernel
VariableV2*
shared_name *$
_class
loc:@pi/dense_2/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ü
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(

pi/dense_2/kernel/readIdentitypi/dense_2/kernel*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel

!pi/dense_2/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@pi/dense_2/bias*
dtype0*
_output_shapes
:

pi/dense_2/bias
VariableV2*
shared_name *"
_class
loc:@pi/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ć
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:

pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

pi/log_std/initial_valueConst*
_output_shapes
:*E
value<B:"0   ż   ż   ż   ż   ż   ż   ż   ż   ż   ż   ż   ż*
dtype0
v

pi/log_std
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ž
pi/log_std/AssignAssign
pi/log_stdpi/log_std/initial_value*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
k
pi/log_std/readIdentity
pi/log_std*
_class
loc:@pi/log_std*
_output_shapes
:*
T0
C
pi/ExpExppi/log_std/read*
_output_shapes
:*
T0
Z
pi/ShapeShapepi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
Z
pi/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
\
pi/random_normal/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

%pi/random_normal/RandomStandardNormalRandomStandardNormalpi/Shape*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2C*

seed *
T0

pi/random_normal/mulMul%pi/random_normal/RandomStandardNormalpi/random_normal/stddev*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
pi/random_normalAddpi/random_normal/mulpi/random_normal/mean*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
pi/mulMulpi/random_normalpi/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
pi/addAddpi/dense_2/BiasAddpi/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
pi/subSubPlaceholder_1pi/dense_2/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
E
pi/Exp_1Exppi/log_std/read*
T0*
_output_shapes
:
O

pi/add_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
J
pi/add_1Addpi/Exp_1
pi/add_1/y*
T0*
_output_shapes
:
Y

pi/truedivRealDivpi/subpi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
M
pi/pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
U
pi/powPow
pi/truedivpi/pow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_1/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
Q
pi/mul_1Mul
pi/mul_1/xpi/log_std/read*
T0*
_output_shapes
:
S
pi/add_2Addpi/powpi/mul_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/add_3/yConst*
valueB
 *?ë?*
dtype0*
_output_shapes
: 
W
pi/add_3Addpi/add_2
pi/add_3/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/mul_2/xConst*
valueB
 *   ż*
dtype0*
_output_shapes
: 
W
pi/mul_2Mul
pi/mul_2/xpi/add_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
pi/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
|
pi/SumSumpi/mul_2pi/Sum/reduction_indices*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0
]
pi/sub_1Subpi/addpi/dense_2/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
E
pi/Exp_2Exppi/log_std/read*
T0*
_output_shapes
:
O

pi/add_4/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
J
pi/add_4Addpi/Exp_2
pi/add_4/y*
_output_shapes
:*
T0
]
pi/truediv_1RealDivpi/sub_1pi/add_4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
[
pi/pow_1Powpi/truediv_1
pi/pow_1/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/mul_3/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
Q
pi/mul_3Mul
pi/mul_3/xpi/log_std/read*
T0*
_output_shapes
:
U
pi/add_5Addpi/pow_1pi/mul_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/add_6/yConst*
valueB
 *?ë?*
dtype0*
_output_shapes
: 
W
pi/add_6Addpi/add_5
pi/add_6/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/mul_4/xConst*
valueB
 *   ż*
dtype0*
_output_shapes
: 
W
pi/mul_4Mul
pi/mul_4/xpi/add_6*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/Sum_1/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0

pi/Sum_1Sumpi/mul_4pi/Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
pi/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
s
pi/Placeholder_1Placeholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_5/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
Q
pi/mul_5Mul
pi/mul_5/xpi/log_std/read*
_output_shapes
:*
T0
>
pi/Exp_3Exppi/mul_5*
T0*
_output_shapes
:
O

pi/mul_6/xConst*
dtype0*
_output_shapes
: *
valueB
 *   @
_
pi/mul_6Mul
pi/mul_6/xpi/Placeholder_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
K
pi/Exp_4Exppi/mul_6*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
pi/sub_2Subpi/Placeholderpi/dense_2/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/pow_2/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
pi/pow_2Powpi/sub_2
pi/pow_2/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
pi/add_7Addpi/pow_2pi/Exp_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/add_8/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
W
pi/add_8Addpi/Exp_4
pi/add_8/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
pi/truediv_2RealDivpi/add_7pi/add_8*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/sub_3/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
[
pi/sub_3Subpi/truediv_2
pi/sub_3/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_7/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
W
pi/mul_7Mul
pi/mul_7/xpi/sub_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
]
pi/add_9Addpi/mul_7pi/Placeholder_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/sub_4Subpi/add_9pi/log_std/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 

pi/Sum_2Sumpi/sub_4pi/Sum_2/reduction_indices*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0
R
pi/ConstConst*
valueB: *
dtype0*
_output_shapes
:
a
pi/MeanMeanpi/Sum_2pi/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
P
pi/add_10/yConst*
valueB
 *Çľ?*
dtype0*
_output_shapes
: 
S
	pi/add_10Addpi/log_std/readpi/add_10/y*
T0*
_output_shapes
:
e
pi/Sum_3/reduction_indicesConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
t
pi/Sum_3Sum	pi/add_10pi/Sum_3/reduction_indices*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
M

pi/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
e
	pi/Mean_1Meanpi/Sum_3
pi/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Ľ
0vf/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"h      *"
_class
loc:@vf/dense/kernel*
dtype0*
_output_shapes
:

.vf/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *Ľ2ž*"
_class
loc:@vf/dense/kernel*
dtype0

.vf/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *Ľ2>*"
_class
loc:@vf/dense/kernel*
dtype0
đ
8vf/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vf/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	h*

seed *
T0*"
_class
loc:@vf/dense/kernel*
seed2
Ú
.vf/dense/kernel/Initializer/random_uniform/subSub.vf/dense/kernel/Initializer/random_uniform/max.vf/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
: 
í
.vf/dense/kernel/Initializer/random_uniform/mulMul8vf/dense/kernel/Initializer/random_uniform/RandomUniform.vf/dense/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	h
ß
*vf/dense/kernel/Initializer/random_uniformAdd.vf/dense/kernel/Initializer/random_uniform/mul.vf/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	h*
T0*"
_class
loc:@vf/dense/kernel
Š
vf/dense/kernel
VariableV2*
shared_name *"
_class
loc:@vf/dense/kernel*
	container *
shape:	h*
dtype0*
_output_shapes
:	h
Ô
vf/dense/kernel/AssignAssignvf/dense/kernel*vf/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	h

vf/dense/kernel/readIdentityvf/dense/kernel*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	h

vf/dense/bias/Initializer/zerosConst*
valueB*    * 
_class
loc:@vf/dense/bias*
dtype0*
_output_shapes	
:

vf/dense/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name * 
_class
loc:@vf/dense/bias*
	container 
ż
vf/dense/bias/AssignAssignvf/dense/biasvf/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vf/dense/bias
u
vf/dense/bias/readIdentityvf/dense/bias*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:

vf/dense/MatMulMatMulPlaceholdervf/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

vf/dense/BiasAddBiasAddvf/dense/MatMulvf/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
vf/dense/TanhTanhvf/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2vf/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *$
_class
loc:@vf/dense_1/kernel*
dtype0*
_output_shapes
:

0vf/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *×łÝ˝*$
_class
loc:@vf/dense_1/kernel*
dtype0*
_output_shapes
: 

0vf/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *×łÝ=*$
_class
loc:@vf/dense_1/kernel
÷
:vf/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*$
_class
loc:@vf/dense_1/kernel*
seed2
â
0vf/dense_1/kernel/Initializer/random_uniform/subSub0vf/dense_1/kernel/Initializer/random_uniform/max0vf/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@vf/dense_1/kernel*
_output_shapes
: *
T0
ö
0vf/dense_1/kernel/Initializer/random_uniform/mulMul:vf/dense_1/kernel/Initializer/random_uniform/RandomUniform0vf/dense_1/kernel/Initializer/random_uniform/sub*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0
č
,vf/dense_1/kernel/Initializer/random_uniformAdd0vf/dense_1/kernel/Initializer/random_uniform/mul0vf/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel
Ż
vf/dense_1/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *$
_class
loc:@vf/dense_1/kernel*
	container *
shape:

Ý
vf/dense_1/kernel/AssignAssignvf/dense_1/kernel,vf/dense_1/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel

vf/dense_1/kernel/readIdentityvf/dense_1/kernel*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:


!vf/dense_1/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@vf/dense_1/bias*
dtype0*
_output_shapes	
:
Ą
vf/dense_1/bias
VariableV2*"
_class
loc:@vf/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ç
vf/dense_1/bias/AssignAssignvf/dense_1/bias!vf/dense_1/bias/Initializer/zeros*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
{
vf/dense_1/bias/readIdentityvf/dense_1/bias*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:

vf/dense_1/MatMulMatMulvf/dense/Tanhvf/dense_1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

vf/dense_1/BiasAddBiasAddvf/dense_1/MatMulvf/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
^
vf/dense_1/TanhTanhvf/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2vf/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *$
_class
loc:@vf/dense_2/kernel*
dtype0*
_output_shapes
:

0vf/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *Ivž*$
_class
loc:@vf/dense_2/kernel*
dtype0*
_output_shapes
: 

0vf/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Iv>*$
_class
loc:@vf/dense_2/kernel*
dtype0*
_output_shapes
: 
ö
:vf/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_2/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@vf/dense_2/kernel*
seed2Ź*
dtype0*
_output_shapes
:	*

seed 
â
0vf/dense_2/kernel/Initializer/random_uniform/subSub0vf/dense_2/kernel/Initializer/random_uniform/max0vf/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
: 
ő
0vf/dense_2/kernel/Initializer/random_uniform/mulMul:vf/dense_2/kernel/Initializer/random_uniform/RandomUniform0vf/dense_2/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
ç
,vf/dense_2/kernel/Initializer/random_uniformAdd0vf/dense_2/kernel/Initializer/random_uniform/mul0vf/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
­
vf/dense_2/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *$
_class
loc:@vf/dense_2/kernel*
	container *
shape:	
Ü
vf/dense_2/kernel/AssignAssignvf/dense_2/kernel,vf/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	

vf/dense_2/kernel/readIdentityvf/dense_2/kernel*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	

!vf/dense_2/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@vf/dense_2/bias*
dtype0*
_output_shapes
:

vf/dense_2/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@vf/dense_2/bias
Ć
vf/dense_2/bias/AssignAssignvf/dense_2/bias!vf/dense_2/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:
z
vf/dense_2/bias/readIdentityvf/dense_2/bias*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:

vf/dense_2/MatMulMatMulvf/dense_1/Tanhvf/dense_2/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

vf/dense_2/BiasAddBiasAddvf/dense_2/MatMulvf/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n

vf/SqueezeSqueezevf/dense_2/BiasAdd*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
*
T0
Ľ
0vc/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"h      *"
_class
loc:@vc/dense/kernel*
dtype0*
_output_shapes
:

.vc/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *Ľ2ž*"
_class
loc:@vc/dense/kernel

.vc/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *Ľ2>*"
_class
loc:@vc/dense/kernel
đ
8vc/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vc/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	h*

seed *
T0*"
_class
loc:@vc/dense/kernel*
seed2˝
Ú
.vc/dense/kernel/Initializer/random_uniform/subSub.vc/dense/kernel/Initializer/random_uniform/max.vc/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@vc/dense/kernel
í
.vc/dense/kernel/Initializer/random_uniform/mulMul8vc/dense/kernel/Initializer/random_uniform/RandomUniform.vc/dense/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	h
ß
*vc/dense/kernel/Initializer/random_uniformAdd.vc/dense/kernel/Initializer/random_uniform/mul.vc/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	h
Š
vc/dense/kernel
VariableV2*
dtype0*
_output_shapes
:	h*
shared_name *"
_class
loc:@vc/dense/kernel*
	container *
shape:	h
Ô
vc/dense/kernel/AssignAssignvc/dense/kernel*vc/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	h

vc/dense/kernel/readIdentityvc/dense/kernel*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	h*
T0

vc/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    * 
_class
loc:@vc/dense/bias

vc/dense/bias
VariableV2*
shared_name * 
_class
loc:@vc/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ż
vc/dense/bias/AssignAssignvc/dense/biasvc/dense/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:
u
vc/dense/bias/readIdentityvc/dense/bias*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:

vc/dense/MatMulMatMulPlaceholdervc/dense/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

vc/dense/BiasAddBiasAddvc/dense/MatMulvc/dense/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Z
vc/dense/TanhTanhvc/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2vc/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *$
_class
loc:@vc/dense_1/kernel*
dtype0

0vc/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *×łÝ˝*$
_class
loc:@vc/dense_1/kernel*
dtype0

0vc/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*$
_class
loc:@vc/dense_1/kernel*
dtype0*
_output_shapes
: 
÷
:vc/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_1/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*

seed *
T0*$
_class
loc:@vc/dense_1/kernel*
seed2Î*
dtype0
â
0vc/dense_1/kernel/Initializer/random_uniform/subSub0vc/dense_1/kernel/Initializer/random_uniform/max0vc/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
: 
ö
0vc/dense_1/kernel/Initializer/random_uniform/mulMul:vc/dense_1/kernel/Initializer/random_uniform/RandomUniform0vc/dense_1/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

č
,vc/dense_1/kernel/Initializer/random_uniformAdd0vc/dense_1/kernel/Initializer/random_uniform/mul0vc/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel
Ż
vc/dense_1/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *$
_class
loc:@vc/dense_1/kernel
Ý
vc/dense_1/kernel/AssignAssignvc/dense_1/kernel,vc/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:


vc/dense_1/kernel/readIdentityvc/dense_1/kernel*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0

!vc/dense_1/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@vc/dense_1/bias*
dtype0*
_output_shapes	
:
Ą
vc/dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@vc/dense_1/bias*
	container *
shape:
Ç
vc/dense_1/bias/AssignAssignvc/dense_1/bias!vc/dense_1/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:
{
vc/dense_1/bias/readIdentityvc/dense_1/bias*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:

vc/dense_1/MatMulMatMulvc/dense/Tanhvc/dense_1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

vc/dense_1/BiasAddBiasAddvc/dense_1/MatMulvc/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
vc/dense_1/TanhTanhvc/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2vc/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *$
_class
loc:@vc/dense_2/kernel*
dtype0*
_output_shapes
:

0vc/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *Ivž*$
_class
loc:@vc/dense_2/kernel*
dtype0*
_output_shapes
: 

0vc/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Iv>*$
_class
loc:@vc/dense_2/kernel*
dtype0*
_output_shapes
: 
ö
:vc/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*$
_class
loc:@vc/dense_2/kernel*
seed2ß
â
0vc/dense_2/kernel/Initializer/random_uniform/subSub0vc/dense_2/kernel/Initializer/random_uniform/max0vc/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
: 
ő
0vc/dense_2/kernel/Initializer/random_uniform/mulMul:vc/dense_2/kernel/Initializer/random_uniform/RandomUniform0vc/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel
ç
,vc/dense_2/kernel/Initializer/random_uniformAdd0vc/dense_2/kernel/Initializer/random_uniform/mul0vc/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	
­
vc/dense_2/kernel
VariableV2*
shared_name *$
_class
loc:@vc/dense_2/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ü
vc/dense_2/kernel/AssignAssignvc/dense_2/kernel,vc/dense_2/kernel/Initializer/random_uniform*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0

vc/dense_2/kernel/readIdentityvc/dense_2/kernel*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	

!vc/dense_2/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@vc/dense_2/bias*
dtype0*
_output_shapes
:

vc/dense_2/bias
VariableV2*"
_class
loc:@vc/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ć
vc/dense_2/bias/AssignAssignvc/dense_2/bias!vc/dense_2/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:
z
vc/dense_2/bias/readIdentityvc/dense_2/bias*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:

vc/dense_2/MatMulMatMulvc/dense_1/Tanhvc/dense_2/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

vc/dense_2/BiasAddBiasAddvc/dense_2/MatMulvc/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
n

vc/SqueezeSqueezevc/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
@
NegNegpi/Sum*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
ConstConst*
_output_shapes
:*
valueB: *
dtype0
V
MeanMeanNegConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
Const_1Const*
valueB
 *D
?*
dtype0*
_output_shapes
: 
h
#penalty/penalty_param/initial_valueConst*
_output_shapes
: *
valueB
 *D
?*
dtype0
y
penalty/penalty_param
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
Ö
penalty/penalty_param/AssignAssignpenalty/penalty_param#penalty/penalty_param/initial_value*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 

penalty/penalty_param/readIdentitypenalty/penalty_param*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Q
SoftplusSoftpluspenalty/penalty_param/read*
T0*
_output_shapes
: 
I
Neg_1Negpenalty/penalty_param/read*
_output_shapes
: *
T0
J
sub/yConst*
valueB
 *  ČA*
dtype0*
_output_shapes
: 
A
subSubPlaceholder_8sub/y*
_output_shapes
: *
T0
7
mulMulNeg_1sub*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
S
gradients/mul_grad/MulMulgradients/Fillsub*
T0*
_output_shapes
: 
W
gradients/mul_grad/Mul_1Mulgradients/FillNeg_1*
_output_shapes
: *
T0
_
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Mul^gradients/mul_grad/Mul_1
Á
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Mul$^gradients/mul_grad/tuple/group_deps*)
_class
loc:@gradients/mul_grad/Mul*
_output_shapes
: *
T0
Ç
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Mul_1$^gradients/mul_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_grad/Mul_1*
_output_shapes
: 
m
gradients/Neg_1_grad/NegNeg+gradients/mul_grad/tuple/control_dependency*
T0*
_output_shapes
: 
`
Reshape/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
n
ReshapeReshapegradients/Neg_1_grad/NegReshape/shape*
T0*
Tshape0*
_output_shapes
:
S
concat/concat_dimConst*
value	B : *
dtype0*
_output_shapes
: 
@
concatIdentityReshape*
T0*
_output_shapes
:
f
PyFuncPyFuncconcat*
_output_shapes
:*
Tin
2*
Tout
2*
token
pyfunc_0
Q
Const_2Const*
_output_shapes
:*
valueB:*
dtype0
Q
split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
s
splitSplitVPyFuncConst_2split/split_dim*
T0*
_output_shapes
:*
	num_split*

Tlen0
R
Reshape_1/shapeConst*
_output_shapes
: *
valueB *
dtype0
[
	Reshape_1ReshapesplitReshape_1/shape*
T0*
Tshape0*
_output_shapes
: 

beta1_power/initial_valueConst*
valueB
 *fff?*(
_class
loc:@penalty/penalty_param*
dtype0*
_output_shapes
: 

beta1_power
VariableV2*
shared_name *(
_class
loc:@penalty/penalty_param*
	container *
shape: *
dtype0*
_output_shapes
: 
¸
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
t
beta1_power/readIdentitybeta1_power*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 

beta2_power/initial_valueConst*
valueB
 *wž?*(
_class
loc:@penalty/penalty_param*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@penalty/penalty_param*
	container *
shape: 
¸
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param
t
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param

,penalty/penalty_param/Adam/Initializer/zerosConst*(
_class
loc:@penalty/penalty_param*
valueB
 *    *
dtype0*
_output_shapes
: 
¨
penalty/penalty_param/Adam
VariableV2*
shared_name *(
_class
loc:@penalty/penalty_param*
	container *
shape: *
dtype0*
_output_shapes
: 
é
!penalty/penalty_param/Adam/AssignAssignpenalty/penalty_param/Adam,penalty/penalty_param/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param

penalty/penalty_param/Adam/readIdentitypenalty/penalty_param/Adam*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param

.penalty/penalty_param/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
valueB
 *    
Ş
penalty/penalty_param/Adam_1
VariableV2*
shared_name *(
_class
loc:@penalty/penalty_param*
	container *
shape: *
dtype0*
_output_shapes
: 
ď
#penalty/penalty_param/Adam_1/AssignAssignpenalty/penalty_param/Adam_1.penalty/penalty_param/Adam_1/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(

!penalty/penalty_param/Adam_1/readIdentitypenalty/penalty_param/Adam_1*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
W
Adam/learning_rateConst*
valueB
 *ÍĚL=*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
ä
+Adam/update_penalty/penalty_param/ApplyAdam	ApplyAdampenalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_1*
_output_shapes
: *
use_locking( *
T0*(
_class
loc:@penalty/penalty_param*
use_nesterov( 
Ś
Adam/mulMulbeta1_power/read
Adam/beta1,^Adam/update_penalty/penalty_param/ApplyAdam*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param
 
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
¨

Adam/mul_1Mulbeta2_power/read
Adam/beta2,^Adam/update_penalty/penalty_param/ApplyAdam*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
¤
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
X
AdamNoOp^Adam/Assign^Adam/Assign_1,^Adam/update_penalty/penalty_param/ApplyAdam
i
Reshape_2/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t
	Reshape_2Reshapepenalty/penalty_param/readReshape_2/shape*
_output_shapes
:*
T0*
Tshape0
\
concat_1/concat_dimConst^Adam*
_output_shapes
: *
value	B : *
dtype0
D
concat_1Identity	Reshape_2*
T0*
_output_shapes
:
h
PyFunc_1PyFuncconcat_1*
_output_shapes
:*
Tin
2*
Tout
2*
token
pyfunc_1
X
Const_3Const^Adam*
valueB:*
dtype0*
_output_shapes
:
Z
split_1/split_dimConst^Adam*
value	B : *
dtype0*
_output_shapes
: 
w
split_1SplitVPyFunc_1Const_3split_1/split_dim*
T0*
_output_shapes
:*
	num_split*

Tlen0
Y
Reshape_3/shapeConst^Adam*
valueB *
dtype0*
_output_shapes
: 
]
	Reshape_3Reshapesplit_1Reshape_3/shape*
T0*
Tshape0*
_output_shapes
: 
Ś
AssignAssignpenalty/penalty_param	Reshape_3*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
"

group_depsNoOp^Adam^Assign
(
group_deps_1NoOp^Adam^group_deps
Q
sub_1Subpi/SumPlaceholder_6*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
?
ExpExpsub_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
	Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
GreaterGreaterPlaceholder_2	Greater/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
L
mul_1/xConst*
valueB
 *?*
dtype0*
_output_shapes
: 
R
mul_1Mulmul_1/xPlaceholder_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
L
mul_2/xConst*
_output_shapes
: *
valueB
 *ÍĚL?*
dtype0
R
mul_2Mulmul_2/xPlaceholder_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
U
SelectSelectGreatermul_1mul_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
mul_3MulExpPlaceholder_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
MinimumMinimummul_3Select*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:
^
Mean_1MeanMinimumConst_4*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
N
mul_4MulExpPlaceholder_3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_5Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_2Meanmul_4Const_5*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
L
mul_5/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
A
mul_5Mulmul_5/x	pi/Mean_1*
T0*
_output_shapes
: 
:
addAddMean_1mul_5*
_output_shapes
: *
T0
?
mul_6MulSoftplusMean_2*
T0*
_output_shapes
: 
9
sub_2Subaddmul_6*
T0*
_output_shapes
: 
L
add_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
@
add_1Addadd_1/xSoftplus*
_output_shapes
: *
T0
A
truedivRealDivsub_2add_1*
_output_shapes
: *
T0
6
Neg_2Negtruediv*
T0*
_output_shapes
: 
T
gradients_1/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Z
gradients_1/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
_output_shapes
: *
T0*

index_type0
T
gradients_1/Neg_2_grad/NegNeggradients_1/Fill*
_output_shapes
: *
T0
a
gradients_1/truediv_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
c
 gradients_1/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Ć
.gradients_1/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/truediv_grad/Shape gradients_1/truediv_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
o
 gradients_1/truediv_grad/RealDivRealDivgradients_1/Neg_2_grad/Negadd_1*
T0*
_output_shapes
: 
ł
gradients_1/truediv_grad/SumSum gradients_1/truediv_grad/RealDiv.gradients_1/truediv_grad/BroadcastGradientArgs*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

 gradients_1/truediv_grad/ReshapeReshapegradients_1/truediv_grad/Sumgradients_1/truediv_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
K
gradients_1/truediv_grad/NegNegsub_2*
T0*
_output_shapes
: 
s
"gradients_1/truediv_grad/RealDiv_1RealDivgradients_1/truediv_grad/Negadd_1*
_output_shapes
: *
T0
y
"gradients_1/truediv_grad/RealDiv_2RealDiv"gradients_1/truediv_grad/RealDiv_1add_1*
T0*
_output_shapes
: 

gradients_1/truediv_grad/mulMulgradients_1/Neg_2_grad/Neg"gradients_1/truediv_grad/RealDiv_2*
_output_shapes
: *
T0
ł
gradients_1/truediv_grad/Sum_1Sumgradients_1/truediv_grad/mul0gradients_1/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

"gradients_1/truediv_grad/Reshape_1Reshapegradients_1/truediv_grad/Sum_1 gradients_1/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
y
)gradients_1/truediv_grad/tuple/group_depsNoOp!^gradients_1/truediv_grad/Reshape#^gradients_1/truediv_grad/Reshape_1
á
1gradients_1/truediv_grad/tuple/control_dependencyIdentity gradients_1/truediv_grad/Reshape*^gradients_1/truediv_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/truediv_grad/Reshape*
_output_shapes
: *
T0
ç
3gradients_1/truediv_grad/tuple/control_dependency_1Identity"gradients_1/truediv_grad/Reshape_1*^gradients_1/truediv_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1*
_output_shapes
: 
u
gradients_1/sub_2_grad/NegNeg1gradients_1/truediv_grad/tuple/control_dependency*
T0*
_output_shapes
: 

'gradients_1/sub_2_grad/tuple/group_depsNoOp^gradients_1/sub_2_grad/Neg2^gradients_1/truediv_grad/tuple/control_dependency
î
/gradients_1/sub_2_grad/tuple/control_dependencyIdentity1gradients_1/truediv_grad/tuple/control_dependency(^gradients_1/sub_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/truediv_grad/Reshape*
_output_shapes
: 
Ó
1gradients_1/sub_2_grad/tuple/control_dependency_1Identitygradients_1/sub_2_grad/Neg(^gradients_1/sub_2_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/sub_2_grad/Neg*
_output_shapes
: 
e
'gradients_1/add_1_grad/tuple/group_depsNoOp4^gradients_1/truediv_grad/tuple/control_dependency_1
ň
/gradients_1/add_1_grad/tuple/control_dependencyIdentity3gradients_1/truediv_grad/tuple/control_dependency_1(^gradients_1/add_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1*
_output_shapes
: 
ô
1gradients_1/add_1_grad/tuple/control_dependency_1Identity3gradients_1/truediv_grad/tuple/control_dependency_1(^gradients_1/add_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1*
_output_shapes
: 
_
%gradients_1/add_grad/tuple/group_depsNoOp0^gradients_1/sub_2_grad/tuple/control_dependency
č
-gradients_1/add_grad/tuple/control_dependencyIdentity/gradients_1/sub_2_grad/tuple/control_dependency&^gradients_1/add_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/truediv_grad/Reshape*
_output_shapes
: 
ę
/gradients_1/add_grad/tuple/control_dependency_1Identity/gradients_1/sub_2_grad/tuple/control_dependency&^gradients_1/add_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_1/truediv_grad/Reshape
}
gradients_1/mul_6_grad/MulMul1gradients_1/sub_2_grad/tuple/control_dependency_1Mean_2*
T0*
_output_shapes
: 

gradients_1/mul_6_grad/Mul_1Mul1gradients_1/sub_2_grad/tuple/control_dependency_1Softplus*
T0*
_output_shapes
: 
k
'gradients_1/mul_6_grad/tuple/group_depsNoOp^gradients_1/mul_6_grad/Mul^gradients_1/mul_6_grad/Mul_1
Ń
/gradients_1/mul_6_grad/tuple/control_dependencyIdentitygradients_1/mul_6_grad/Mul(^gradients_1/mul_6_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_6_grad/Mul*
_output_shapes
: 
×
1gradients_1/mul_6_grad/tuple/control_dependency_1Identitygradients_1/mul_6_grad/Mul_1(^gradients_1/mul_6_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/mul_6_grad/Mul_1*
_output_shapes
: 
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ł
gradients_1/Mean_1_grad/ReshapeReshape-gradients_1/add_grad/tuple/control_dependency%gradients_1/Mean_1_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
d
gradients_1/Mean_1_grad/ShapeShapeMinimum*
T0*
out_type0*
_output_shapes
:
¤
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
gradients_1/Mean_1_grad/Shape_1ShapeMinimum*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
˘
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_1/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ś
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0*
_output_shapes
: 

gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0

gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
gradients_1/mul_5_grad/MulMul/gradients_1/add_grad/tuple/control_dependency_1	pi/Mean_1*
T0*
_output_shapes
: 
~
gradients_1/mul_5_grad/Mul_1Mul/gradients_1/add_grad/tuple/control_dependency_1mul_5/x*
T0*
_output_shapes
: 
k
'gradients_1/mul_5_grad/tuple/group_depsNoOp^gradients_1/mul_5_grad/Mul^gradients_1/mul_5_grad/Mul_1
Ń
/gradients_1/mul_5_grad/tuple/control_dependencyIdentitygradients_1/mul_5_grad/Mul(^gradients_1/mul_5_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_5_grad/Mul*
_output_shapes
: 
×
1gradients_1/mul_5_grad/tuple/control_dependency_1Identitygradients_1/mul_5_grad/Mul_1(^gradients_1/mul_5_grad/tuple/group_deps*/
_class%
#!loc:@gradients_1/mul_5_grad/Mul_1*
_output_shapes
: *
T0
Ý
gradients_1/AddNAddN1gradients_1/add_1_grad/tuple/control_dependency_1/gradients_1/mul_6_grad/tuple/control_dependency*
_output_shapes
: *
T0*5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1*
N
i
!gradients_1/Softplus_grad/SigmoidSigmoidpenalty/penalty_param/read*
T0*
_output_shapes
: 
z
gradients_1/Softplus_grad/mulMulgradients_1/AddN!gradients_1/Softplus_grad/Sigmoid*
T0*
_output_shapes
: 
o
%gradients_1/Mean_2_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
ˇ
gradients_1/Mean_2_grad/ReshapeReshape1gradients_1/mul_6_grad/tuple/control_dependency_1%gradients_1/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
b
gradients_1/Mean_2_grad/ShapeShapemul_4*
_output_shapes
:*
T0*
out_type0
¤
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
d
gradients_1/Mean_2_grad/Shape_1Shapemul_4*
_output_shapes
:*
T0*
out_type0
b
gradients_1/Mean_2_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_2_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
˘
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
gradients_1/Mean_2_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Ś
gradients_1/Mean_2_grad/Prod_1Prodgradients_1/Mean_2_grad/Shape_2gradients_1/Mean_2_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
c
!gradients_1/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0

gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
_output_shapes
: *
T0

gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c
gradients_1/Minimum_grad/ShapeShapemul_3*
T0*
out_type0*
_output_shapes
:
f
 gradients_1/Minimum_grad/Shape_1ShapeSelect*
T0*
out_type0*
_output_shapes
:

 gradients_1/Minimum_grad/Shape_2Shapegradients_1/Mean_1_grad/truediv*
out_type0*
_output_shapes
:*
T0
i
$gradients_1/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ž
gradients_1/Minimum_grad/zerosFill gradients_1/Minimum_grad/Shape_2$gradients_1/Minimum_grad/zeros/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
"gradients_1/Minimum_grad/LessEqual	LessEqualmul_3Select*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ć
.gradients_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Minimum_grad/Shape gradients_1/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ź
gradients_1/Minimum_grad/SelectSelect"gradients_1/Minimum_grad/LessEqualgradients_1/Mean_1_grad/truedivgradients_1/Minimum_grad/zeros*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients_1/Minimum_grad/SumSumgradients_1/Minimum_grad/Select.gradients_1/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ľ
 gradients_1/Minimum_grad/ReshapeReshapegradients_1/Minimum_grad/Sumgradients_1/Minimum_grad/Shape*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
!gradients_1/Minimum_grad/Select_1Select"gradients_1/Minimum_grad/LessEqualgradients_1/Minimum_grad/zerosgradients_1/Mean_1_grad/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
gradients_1/Minimum_grad/Sum_1Sum!gradients_1/Minimum_grad/Select_10gradients_1/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ť
"gradients_1/Minimum_grad/Reshape_1Reshapegradients_1/Minimum_grad/Sum_1 gradients_1/Minimum_grad/Shape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
y
)gradients_1/Minimum_grad/tuple/group_depsNoOp!^gradients_1/Minimum_grad/Reshape#^gradients_1/Minimum_grad/Reshape_1
î
1gradients_1/Minimum_grad/tuple/control_dependencyIdentity gradients_1/Minimum_grad/Reshape*^gradients_1/Minimum_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*3
_class)
'%loc:@gradients_1/Minimum_grad/Reshape
ô
3gradients_1/Minimum_grad/tuple/control_dependency_1Identity"gradients_1/Minimum_grad/Reshape_1*^gradients_1/Minimum_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/Minimum_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
(gradients_1/pi/Mean_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
š
"gradients_1/pi/Mean_1_grad/ReshapeReshape1gradients_1/mul_5_grad/tuple/control_dependency_1(gradients_1/pi/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
c
 gradients_1/pi/Mean_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
 
gradients_1/pi/Mean_1_grad/TileTile"gradients_1/pi/Mean_1_grad/Reshape gradients_1/pi/Mean_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
g
"gradients_1/pi/Mean_1_grad/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"gradients_1/pi/Mean_1_grad/truedivRealDivgradients_1/pi/Mean_1_grad/Tile"gradients_1/pi/Mean_1_grad/Const_1*
T0*
_output_shapes
: 
_
gradients_1/mul_4_grad/ShapeShapeExp*
T0*
out_type0*
_output_shapes
:
k
gradients_1/mul_4_grad/Shape_1ShapePlaceholder_3*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_1/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_4_grad/Shapegradients_1/mul_4_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients_1/mul_4_grad/MulMulgradients_1/Mean_2_grad/truedivPlaceholder_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
gradients_1/mul_4_grad/SumSumgradients_1/mul_4_grad/Mul,gradients_1/mul_4_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/mul_4_grad/ReshapeReshapegradients_1/mul_4_grad/Sumgradients_1/mul_4_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
gradients_1/mul_4_grad/Mul_1MulExpgradients_1/Mean_2_grad/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ą
gradients_1/mul_4_grad/Sum_1Sumgradients_1/mul_4_grad/Mul_1.gradients_1/mul_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ľ
 gradients_1/mul_4_grad/Reshape_1Reshapegradients_1/mul_4_grad/Sum_1gradients_1/mul_4_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_1/mul_4_grad/tuple/group_depsNoOp^gradients_1/mul_4_grad/Reshape!^gradients_1/mul_4_grad/Reshape_1
ć
/gradients_1/mul_4_grad/tuple/control_dependencyIdentitygradients_1/mul_4_grad/Reshape(^gradients_1/mul_4_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*1
_class'
%#loc:@gradients_1/mul_4_grad/Reshape
ě
1gradients_1/mul_4_grad/tuple/control_dependency_1Identity gradients_1/mul_4_grad/Reshape_1(^gradients_1/mul_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/mul_4_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_1/mul_3_grad/ShapeShapeExp*
_output_shapes
:*
T0*
out_type0
k
gradients_1/mul_3_grad/Shape_1ShapePlaceholder_2*
_output_shapes
:*
T0*
out_type0
Ŕ
,gradients_1/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_3_grad/Shapegradients_1/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients_1/mul_3_grad/MulMul1gradients_1/Minimum_grad/tuple/control_dependencyPlaceholder_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
gradients_1/mul_3_grad/SumSumgradients_1/mul_3_grad/Mul,gradients_1/mul_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients_1/mul_3_grad/ReshapeReshapegradients_1/mul_3_grad/Sumgradients_1/mul_3_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/mul_3_grad/Mul_1MulExp1gradients_1/Minimum_grad/tuple/control_dependency*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients_1/mul_3_grad/Sum_1Sumgradients_1/mul_3_grad/Mul_1.gradients_1/mul_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ľ
 gradients_1/mul_3_grad/Reshape_1Reshapegradients_1/mul_3_grad/Sum_1gradients_1/mul_3_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_1/mul_3_grad/tuple/group_depsNoOp^gradients_1/mul_3_grad/Reshape!^gradients_1/mul_3_grad/Reshape_1
ć
/gradients_1/mul_3_grad/tuple/control_dependencyIdentitygradients_1/mul_3_grad/Reshape(^gradients_1/mul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/mul_3_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
1gradients_1/mul_3_grad/tuple/control_dependency_1Identity gradients_1/mul_3_grad/Reshape_1(^gradients_1/mul_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/mul_3_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
gradients_1/pi/Sum_3_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients_1/pi/Sum_3_grad/SizeConst*
value	B :*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
ľ
gradients_1/pi/Sum_3_grad/addAddpi/Sum_3/reduction_indicesgradients_1/pi/Sum_3_grad/Size*
T0*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
_output_shapes
: 
˝
gradients_1/pi/Sum_3_grad/modFloorModgradients_1/pi/Sum_3_grad/addgradients_1/pi/Sum_3_grad/Size*
T0*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
_output_shapes
: 

!gradients_1/pi/Sum_3_grad/Shape_1Const*
valueB *2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 

%gradients_1/pi/Sum_3_grad/range/startConst*
value	B : *2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 

%gradients_1/pi/Sum_3_grad/range/deltaConst*
value	B :*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
ň
gradients_1/pi/Sum_3_grad/rangeRange%gradients_1/pi/Sum_3_grad/range/startgradients_1/pi/Sum_3_grad/Size%gradients_1/pi/Sum_3_grad/range/delta*
_output_shapes
:*

Tidx0*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape

$gradients_1/pi/Sum_3_grad/Fill/valueConst*
value	B :*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
Ö
gradients_1/pi/Sum_3_grad/FillFill!gradients_1/pi/Sum_3_grad/Shape_1$gradients_1/pi/Sum_3_grad/Fill/value*
T0*

index_type0*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
_output_shapes
: 

'gradients_1/pi/Sum_3_grad/DynamicStitchDynamicStitchgradients_1/pi/Sum_3_grad/rangegradients_1/pi/Sum_3_grad/modgradients_1/pi/Sum_3_grad/Shapegradients_1/pi/Sum_3_grad/Fill*
T0*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
N*
_output_shapes
:

#gradients_1/pi/Sum_3_grad/Maximum/yConst*
value	B :*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
Ó
!gradients_1/pi/Sum_3_grad/MaximumMaximum'gradients_1/pi/Sum_3_grad/DynamicStitch#gradients_1/pi/Sum_3_grad/Maximum/y*
T0*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
_output_shapes
:
Ë
"gradients_1/pi/Sum_3_grad/floordivFloorDivgradients_1/pi/Sum_3_grad/Shape!gradients_1/pi/Sum_3_grad/Maximum*
T0*2
_class(
&$loc:@gradients_1/pi/Sum_3_grad/Shape*
_output_shapes
:
Ź
!gradients_1/pi/Sum_3_grad/ReshapeReshape"gradients_1/pi/Mean_1_grad/truediv'gradients_1/pi/Sum_3_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
¤
gradients_1/pi/Sum_3_grad/TileTile!gradients_1/pi/Sum_3_grad/Reshape"gradients_1/pi/Sum_3_grad/floordiv*
T0*
_output_shapes
:*

Tmultiples0
ć
gradients_1/AddN_1AddN/gradients_1/mul_4_grad/tuple/control_dependency/gradients_1/mul_3_grad/tuple/control_dependency*
T0*1
_class'
%#loc:@gradients_1/mul_4_grad/Reshape*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
gradients_1/Exp_grad/mulMulgradients_1/AddN_1Exp*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
 gradients_1/pi/add_10_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
e
"gradients_1/pi/add_10_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Ě
0gradients_1/pi/add_10_grad/BroadcastGradientArgsBroadcastGradientArgs gradients_1/pi/add_10_grad/Shape"gradients_1/pi/add_10_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
š
gradients_1/pi/add_10_grad/SumSumgradients_1/pi/Sum_3_grad/Tile0gradients_1/pi/add_10_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
˘
"gradients_1/pi/add_10_grad/ReshapeReshapegradients_1/pi/add_10_grad/Sum gradients_1/pi/add_10_grad/Shape*
_output_shapes
:*
T0*
Tshape0
š
 gradients_1/pi/add_10_grad/Sum_1Sumgradients_1/pi/Sum_3_grad/Tile2gradients_1/pi/add_10_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
¤
$gradients_1/pi/add_10_grad/Reshape_1Reshape gradients_1/pi/add_10_grad/Sum_1"gradients_1/pi/add_10_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

+gradients_1/pi/add_10_grad/tuple/group_depsNoOp#^gradients_1/pi/add_10_grad/Reshape%^gradients_1/pi/add_10_grad/Reshape_1
í
3gradients_1/pi/add_10_grad/tuple/control_dependencyIdentity"gradients_1/pi/add_10_grad/Reshape,^gradients_1/pi/add_10_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/pi/add_10_grad/Reshape*
_output_shapes
:
ď
5gradients_1/pi/add_10_grad/tuple/control_dependency_1Identity$gradients_1/pi/add_10_grad/Reshape_1,^gradients_1/pi/add_10_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients_1/pi/add_10_grad/Reshape_1*
_output_shapes
: 
b
gradients_1/sub_1_grad/ShapeShapepi/Sum*
T0*
out_type0*
_output_shapes
:
k
gradients_1/sub_1_grad/Shape_1ShapePlaceholder_6*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Š
gradients_1/sub_1_grad/SumSumgradients_1/Exp_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
­
gradients_1/sub_1_grad/Sum_1Sumgradients_1/Exp_grad/mul.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
T0*
_output_shapes
:
Ł
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
ć
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
gradients_1/pi/Sum_grad/ShapeShapepi/mul_2*
out_type0*
_output_shapes
:*
T0

gradients_1/pi/Sum_grad/SizeConst*
value	B :*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 
­
gradients_1/pi/Sum_grad/addAddpi/Sum/reduction_indicesgradients_1/pi/Sum_grad/Size*
_output_shapes
: *
T0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape
ľ
gradients_1/pi/Sum_grad/modFloorModgradients_1/pi/Sum_grad/addgradients_1/pi/Sum_grad/Size*
_output_shapes
: *
T0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape

gradients_1/pi/Sum_grad/Shape_1Const*
valueB *0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 

#gradients_1/pi/Sum_grad/range/startConst*
value	B : *0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 

#gradients_1/pi/Sum_grad/range/deltaConst*
value	B :*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 
č
gradients_1/pi/Sum_grad/rangeRange#gradients_1/pi/Sum_grad/range/startgradients_1/pi/Sum_grad/Size#gradients_1/pi/Sum_grad/range/delta*
_output_shapes
:*

Tidx0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape

"gradients_1/pi/Sum_grad/Fill/valueConst*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
dtype0
Î
gradients_1/pi/Sum_grad/FillFillgradients_1/pi/Sum_grad/Shape_1"gradients_1/pi/Sum_grad/Fill/value*
_output_shapes
: *
T0*

index_type0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape

%gradients_1/pi/Sum_grad/DynamicStitchDynamicStitchgradients_1/pi/Sum_grad/rangegradients_1/pi/Sum_grad/modgradients_1/pi/Sum_grad/Shapegradients_1/pi/Sum_grad/Fill*
T0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
N*
_output_shapes
:

!gradients_1/pi/Sum_grad/Maximum/yConst*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
dtype0
Ë
gradients_1/pi/Sum_grad/MaximumMaximum%gradients_1/pi/Sum_grad/DynamicStitch!gradients_1/pi/Sum_grad/Maximum/y*
T0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
_output_shapes
:
Ă
 gradients_1/pi/Sum_grad/floordivFloorDivgradients_1/pi/Sum_grad/Shapegradients_1/pi/Sum_grad/Maximum*
T0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
_output_shapes
:
Ë
gradients_1/pi/Sum_grad/ReshapeReshape/gradients_1/sub_1_grad/tuple/control_dependency%gradients_1/pi/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ť
gradients_1/pi/Sum_grad/TileTilegradients_1/pi/Sum_grad/Reshape gradients_1/pi/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients_1/pi/mul_2_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
i
!gradients_1/pi/mul_2_grad/Shape_1Shapepi/add_3*
_output_shapes
:*
T0*
out_type0
É
/gradients_1/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/mul_2_grad/Shape!gradients_1/pi/mul_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
~
gradients_1/pi/mul_2_grad/MulMulgradients_1/pi/Sum_grad/Tilepi/add_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
´
gradients_1/pi/mul_2_grad/SumSumgradients_1/pi/mul_2_grad/Mul/gradients_1/pi/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

!gradients_1/pi/mul_2_grad/ReshapeReshapegradients_1/pi/mul_2_grad/Sumgradients_1/pi/mul_2_grad/Shape*
_output_shapes
: *
T0*
Tshape0

gradients_1/pi/mul_2_grad/Mul_1Mul
pi/mul_2/xgradients_1/pi/Sum_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
gradients_1/pi/mul_2_grad/Sum_1Sumgradients_1/pi/mul_2_grad/Mul_11gradients_1/pi/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
˛
#gradients_1/pi/mul_2_grad/Reshape_1Reshapegradients_1/pi/mul_2_grad/Sum_1!gradients_1/pi/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
*gradients_1/pi/mul_2_grad/tuple/group_depsNoOp"^gradients_1/pi/mul_2_grad/Reshape$^gradients_1/pi/mul_2_grad/Reshape_1
ĺ
2gradients_1/pi/mul_2_grad/tuple/control_dependencyIdentity!gradients_1/pi/mul_2_grad/Reshape+^gradients_1/pi/mul_2_grad/tuple/group_deps*
_output_shapes
: *
T0*4
_class*
(&loc:@gradients_1/pi/mul_2_grad/Reshape
ü
4gradients_1/pi/mul_2_grad/tuple/control_dependency_1Identity#gradients_1/pi/mul_2_grad/Reshape_1+^gradients_1/pi/mul_2_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/pi/mul_2_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
gradients_1/pi/add_3_grad/ShapeShapepi/add_2*
T0*
out_type0*
_output_shapes
:
d
!gradients_1/pi/add_3_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
É
/gradients_1/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_3_grad/Shape!gradients_1/pi/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ë
gradients_1/pi/add_3_grad/SumSum4gradients_1/pi/mul_2_grad/tuple/control_dependency_1/gradients_1/pi/add_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ź
!gradients_1/pi/add_3_grad/ReshapeReshapegradients_1/pi/add_3_grad/Sumgradients_1/pi/add_3_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ď
gradients_1/pi/add_3_grad/Sum_1Sum4gradients_1/pi/mul_2_grad/tuple/control_dependency_11gradients_1/pi/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ą
#gradients_1/pi/add_3_grad/Reshape_1Reshapegradients_1/pi/add_3_grad/Sum_1!gradients_1/pi/add_3_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
|
*gradients_1/pi/add_3_grad/tuple/group_depsNoOp"^gradients_1/pi/add_3_grad/Reshape$^gradients_1/pi/add_3_grad/Reshape_1
ö
2gradients_1/pi/add_3_grad/tuple/control_dependencyIdentity!gradients_1/pi/add_3_grad/Reshape+^gradients_1/pi/add_3_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/pi/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
4gradients_1/pi/add_3_grad/tuple/control_dependency_1Identity#gradients_1/pi/add_3_grad/Reshape_1+^gradients_1/pi/add_3_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/pi/add_3_grad/Reshape_1*
_output_shapes
: 
e
gradients_1/pi/add_2_grad/ShapeShapepi/pow*
T0*
out_type0*
_output_shapes
:
k
!gradients_1/pi/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
É
/gradients_1/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_2_grad/Shape!gradients_1/pi/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
gradients_1/pi/add_2_grad/SumSum2gradients_1/pi/add_3_grad/tuple/control_dependency/gradients_1/pi/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ź
!gradients_1/pi/add_2_grad/ReshapeReshapegradients_1/pi/add_2_grad/Sumgradients_1/pi/add_2_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Í
gradients_1/pi/add_2_grad/Sum_1Sum2gradients_1/pi/add_3_grad/tuple/control_dependency1gradients_1/pi/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ľ
#gradients_1/pi/add_2_grad/Reshape_1Reshapegradients_1/pi/add_2_grad/Sum_1!gradients_1/pi/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
|
*gradients_1/pi/add_2_grad/tuple/group_depsNoOp"^gradients_1/pi/add_2_grad/Reshape$^gradients_1/pi/add_2_grad/Reshape_1
ö
2gradients_1/pi/add_2_grad/tuple/control_dependencyIdentity!gradients_1/pi/add_2_grad/Reshape+^gradients_1/pi/add_2_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*4
_class*
(&loc:@gradients_1/pi/add_2_grad/Reshape
ď
4gradients_1/pi/add_2_grad/tuple/control_dependency_1Identity#gradients_1/pi/add_2_grad/Reshape_1+^gradients_1/pi/add_2_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/pi/add_2_grad/Reshape_1*
_output_shapes
:
g
gradients_1/pi/pow_grad/ShapeShape
pi/truediv*
T0*
out_type0*
_output_shapes
:
b
gradients_1/pi/pow_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ă
-gradients_1/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/pow_grad/Shapegradients_1/pi/pow_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/pow_grad/mulMul2gradients_1/pi/add_2_grad/tuple/control_dependencypi/pow/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
gradients_1/pi/pow_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
gradients_1/pi/pow_grad/subSubpi/pow/ygradients_1/pi/pow_grad/sub/y*
T0*
_output_shapes
: 
}
gradients_1/pi/pow_grad/PowPow
pi/truedivgradients_1/pi/pow_grad/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/pi/pow_grad/mul_1Mulgradients_1/pi/pow_grad/mulgradients_1/pi/pow_grad/Pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
gradients_1/pi/pow_grad/SumSumgradients_1/pi/pow_grad/mul_1-gradients_1/pi/pow_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ś
gradients_1/pi/pow_grad/ReshapeReshapegradients_1/pi/pow_grad/Sumgradients_1/pi/pow_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
f
!gradients_1/pi/pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients_1/pi/pow_grad/GreaterGreater
pi/truediv!gradients_1/pi/pow_grad/Greater/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
q
'gradients_1/pi/pow_grad/ones_like/ShapeShape
pi/truediv*
T0*
out_type0*
_output_shapes
:
l
'gradients_1/pi/pow_grad/ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ż
!gradients_1/pi/pow_grad/ones_likeFill'gradients_1/pi/pow_grad/ones_like/Shape'gradients_1/pi/pow_grad/ones_like/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
Ş
gradients_1/pi/pow_grad/SelectSelectgradients_1/pi/pow_grad/Greater
pi/truediv!gradients_1/pi/pow_grad/ones_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
gradients_1/pi/pow_grad/LogLoggradients_1/pi/pow_grad/Select*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
"gradients_1/pi/pow_grad/zeros_like	ZerosLike
pi/truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
 gradients_1/pi/pow_grad/Select_1Selectgradients_1/pi/pow_grad/Greatergradients_1/pi/pow_grad/Log"gradients_1/pi/pow_grad/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/pow_grad/mul_2Mul2gradients_1/pi/add_2_grad/tuple/control_dependencypi/pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/pi/pow_grad/mul_3Mulgradients_1/pi/pow_grad/mul_2 gradients_1/pi/pow_grad/Select_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients_1/pi/pow_grad/Sum_1Sumgradients_1/pi/pow_grad/mul_3/gradients_1/pi/pow_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

!gradients_1/pi/pow_grad/Reshape_1Reshapegradients_1/pi/pow_grad/Sum_1gradients_1/pi/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
v
(gradients_1/pi/pow_grad/tuple/group_depsNoOp ^gradients_1/pi/pow_grad/Reshape"^gradients_1/pi/pow_grad/Reshape_1
î
0gradients_1/pi/pow_grad/tuple/control_dependencyIdentitygradients_1/pi/pow_grad/Reshape)^gradients_1/pi/pow_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/pi/pow_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ă
2gradients_1/pi/pow_grad/tuple/control_dependency_1Identity!gradients_1/pi/pow_grad/Reshape_1)^gradients_1/pi/pow_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/pi/pow_grad/Reshape_1*
_output_shapes
: 
b
gradients_1/pi/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
k
!gradients_1/pi/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
É
/gradients_1/pi/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/mul_1_grad/Shape!gradients_1/pi/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients_1/pi/mul_1_grad/MulMul4gradients_1/pi/add_2_grad/tuple/control_dependency_1pi/log_std/read*
T0*
_output_shapes
:
˛
gradients_1/pi/mul_1_grad/SumSumgradients_1/pi/mul_1_grad/Mul/gradients_1/pi/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

!gradients_1/pi/mul_1_grad/ReshapeReshapegradients_1/pi/mul_1_grad/Sumgradients_1/pi/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

gradients_1/pi/mul_1_grad/Mul_1Mul
pi/mul_1/x4gradients_1/pi/add_2_grad/tuple/control_dependency_1*
_output_shapes
:*
T0
ź
gradients_1/pi/mul_1_grad/Sum_1Sumgradients_1/pi/mul_1_grad/Mul_11gradients_1/pi/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ľ
#gradients_1/pi/mul_1_grad/Reshape_1Reshapegradients_1/pi/mul_1_grad/Sum_1!gradients_1/pi/mul_1_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
|
*gradients_1/pi/mul_1_grad/tuple/group_depsNoOp"^gradients_1/pi/mul_1_grad/Reshape$^gradients_1/pi/mul_1_grad/Reshape_1
ĺ
2gradients_1/pi/mul_1_grad/tuple/control_dependencyIdentity!gradients_1/pi/mul_1_grad/Reshape+^gradients_1/pi/mul_1_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/pi/mul_1_grad/Reshape*
_output_shapes
: 
ď
4gradients_1/pi/mul_1_grad/tuple/control_dependency_1Identity#gradients_1/pi/mul_1_grad/Reshape_1+^gradients_1/pi/mul_1_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/pi/mul_1_grad/Reshape_1*
_output_shapes
:
g
!gradients_1/pi/truediv_grad/ShapeShapepi/sub*
T0*
out_type0*
_output_shapes
:
m
#gradients_1/pi/truediv_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ď
1gradients_1/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients_1/pi/truediv_grad/Shape#gradients_1/pi/truediv_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

#gradients_1/pi/truediv_grad/RealDivRealDiv0gradients_1/pi/pow_grad/tuple/control_dependencypi/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
gradients_1/pi/truediv_grad/SumSum#gradients_1/pi/truediv_grad/RealDiv1gradients_1/pi/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
˛
#gradients_1/pi/truediv_grad/ReshapeReshapegradients_1/pi/truediv_grad/Sum!gradients_1/pi/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
gradients_1/pi/truediv_grad/NegNegpi/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

%gradients_1/pi/truediv_grad/RealDiv_1RealDivgradients_1/pi/truediv_grad/Negpi/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

%gradients_1/pi/truediv_grad/RealDiv_2RealDiv%gradients_1/pi/truediv_grad/RealDiv_1pi/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients_1/pi/truediv_grad/mulMul0gradients_1/pi/pow_grad/tuple/control_dependency%gradients_1/pi/truediv_grad/RealDiv_2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
!gradients_1/pi/truediv_grad/Sum_1Sumgradients_1/pi/truediv_grad/mul3gradients_1/pi/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ť
%gradients_1/pi/truediv_grad/Reshape_1Reshape!gradients_1/pi/truediv_grad/Sum_1#gradients_1/pi/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

,gradients_1/pi/truediv_grad/tuple/group_depsNoOp$^gradients_1/pi/truediv_grad/Reshape&^gradients_1/pi/truediv_grad/Reshape_1
ţ
4gradients_1/pi/truediv_grad/tuple/control_dependencyIdentity#gradients_1/pi/truediv_grad/Reshape-^gradients_1/pi/truediv_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*6
_class,
*(loc:@gradients_1/pi/truediv_grad/Reshape
÷
6gradients_1/pi/truediv_grad/tuple/control_dependency_1Identity%gradients_1/pi/truediv_grad/Reshape_1-^gradients_1/pi/truediv_grad/tuple/group_deps*
_output_shapes
:*
T0*8
_class.
,*loc:@gradients_1/pi/truediv_grad/Reshape_1
j
gradients_1/pi/sub_grad/ShapeShapePlaceholder_1*
_output_shapes
:*
T0*
out_type0
q
gradients_1/pi/sub_grad/Shape_1Shapepi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ă
-gradients_1/pi/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/sub_grad/Shapegradients_1/pi/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ç
gradients_1/pi/sub_grad/SumSum4gradients_1/pi/truediv_grad/tuple/control_dependency-gradients_1/pi/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ś
gradients_1/pi/sub_grad/ReshapeReshapegradients_1/pi/sub_grad/Sumgradients_1/pi/sub_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ë
gradients_1/pi/sub_grad/Sum_1Sum4gradients_1/pi/truediv_grad/tuple/control_dependency/gradients_1/pi/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
d
gradients_1/pi/sub_grad/NegNeggradients_1/pi/sub_grad/Sum_1*
_output_shapes
:*
T0
Ş
!gradients_1/pi/sub_grad/Reshape_1Reshapegradients_1/pi/sub_grad/Neggradients_1/pi/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
(gradients_1/pi/sub_grad/tuple/group_depsNoOp ^gradients_1/pi/sub_grad/Reshape"^gradients_1/pi/sub_grad/Reshape_1
î
0gradients_1/pi/sub_grad/tuple/control_dependencyIdentitygradients_1/pi/sub_grad/Reshape)^gradients_1/pi/sub_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*2
_class(
&$loc:@gradients_1/pi/sub_grad/Reshape
ô
2gradients_1/pi/sub_grad/tuple/control_dependency_1Identity!gradients_1/pi/sub_grad/Reshape_1)^gradients_1/pi/sub_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/pi/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
gradients_1/pi/add_1_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
d
!gradients_1/pi/add_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
É
/gradients_1/pi/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_1_grad/Shape!gradients_1/pi/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ď
gradients_1/pi/add_1_grad/SumSum6gradients_1/pi/truediv_grad/tuple/control_dependency_1/gradients_1/pi/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

!gradients_1/pi/add_1_grad/ReshapeReshapegradients_1/pi/add_1_grad/Sumgradients_1/pi/add_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:
Ď
gradients_1/pi/add_1_grad/Sum_1Sum6gradients_1/pi/truediv_grad/tuple/control_dependency_11gradients_1/pi/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Ą
#gradients_1/pi/add_1_grad/Reshape_1Reshapegradients_1/pi/add_1_grad/Sum_1!gradients_1/pi/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients_1/pi/add_1_grad/tuple/group_depsNoOp"^gradients_1/pi/add_1_grad/Reshape$^gradients_1/pi/add_1_grad/Reshape_1
é
2gradients_1/pi/add_1_grad/tuple/control_dependencyIdentity!gradients_1/pi/add_1_grad/Reshape+^gradients_1/pi/add_1_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/pi/add_1_grad/Reshape*
_output_shapes
:
ë
4gradients_1/pi/add_1_grad/tuple/control_dependency_1Identity#gradients_1/pi/add_1_grad/Reshape_1+^gradients_1/pi/add_1_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/pi/add_1_grad/Reshape_1*
_output_shapes
: 
Ž
/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients_1/pi/sub_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
Ł
4gradients_1/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp0^gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad3^gradients_1/pi/sub_grad/tuple/control_dependency_1

<gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity2gradients_1/pi/sub_grad/tuple/control_dependency_15^gradients_1/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/pi/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_1/pi/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*B
_class8
64loc:@gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad

gradients_1/pi/Exp_1_grad/mulMul2gradients_1/pi/add_1_grad/tuple/control_dependencypi/Exp_1*
_output_shapes
:*
T0
â
)gradients_1/pi/dense_2/MatMul_grad/MatMulMatMul<gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependencypi/dense_2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Ô
+gradients_1/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh<gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

3gradients_1/pi/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_1/pi/dense_2/MatMul_grad/MatMul,^gradients_1/pi/dense_2/MatMul_grad/MatMul_1

;gradients_1/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_1/pi/dense_2/MatMul_grad/MatMul4^gradients_1/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_1/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_1/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_1/pi/dense_2/MatMul_grad/MatMul_14^gradients_1/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_1/pi/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	

gradients_1/AddN_2AddN3gradients_1/pi/add_10_grad/tuple/control_dependency4gradients_1/pi/mul_1_grad/tuple/control_dependency_1gradients_1/pi/Exp_1_grad/mul*
T0*5
_class+
)'loc:@gradients_1/pi/add_10_grad/Reshape*
N*
_output_shapes
:
ś
)gradients_1/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh;gradients_1/pi/dense_2/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:*
T0

4gradients_1/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_1/pi/dense_1/Tanh_grad/TanhGrad

<gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_1/pi/dense_1/Tanh_grad/TanhGrad5^gradients_1/pi/dense_1/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*<
_class2
0.loc:@gradients_1/pi/dense_1/Tanh_grad/TanhGrad

>gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_1/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
â
)gradients_1/pi/dense_1/MatMul_grad/MatMulMatMul<gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependencypi/dense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Ó
+gradients_1/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh<gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

3gradients_1/pi/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_1/pi/dense_1/MatMul_grad/MatMul,^gradients_1/pi/dense_1/MatMul_grad/MatMul_1

;gradients_1/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_1/pi/dense_1/MatMul_grad/MatMul4^gradients_1/pi/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*<
_class2
0.loc:@gradients_1/pi/dense_1/MatMul_grad/MatMul

=gradients_1/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_1/pi/dense_1/MatMul_grad/MatMul_14^gradients_1/pi/dense_1/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_1/pi/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:

˛
'gradients_1/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh;gradients_1/pi/dense_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
-gradients_1/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_1/pi/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:*
T0

2gradients_1/pi/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad(^gradients_1/pi/dense/Tanh_grad/TanhGrad

:gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_1/pi/dense/Tanh_grad/TanhGrad3^gradients_1/pi/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*:
_class0
.,loc:@gradients_1/pi/dense/Tanh_grad/TanhGrad

<gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad3^gradients_1/pi/dense/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
Ű
'gradients_1/pi/dense/MatMul_grad/MatMulMatMul:gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h*
transpose_a( *
transpose_b(*
T0
Ě
)gradients_1/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	h*
transpose_a(*
transpose_b( 

1gradients_1/pi/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_1/pi/dense/MatMul_grad/MatMul*^gradients_1/pi/dense/MatMul_grad/MatMul_1

9gradients_1/pi/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_1/pi/dense/MatMul_grad/MatMul2^gradients_1/pi/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_1/pi/dense/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h

;gradients_1/pi/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_1/pi/dense/MatMul_grad/MatMul_12^gradients_1/pi/dense/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_1/pi/dense/MatMul_grad/MatMul_1*
_output_shapes
:	h
b
Reshape_4/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_4Reshape;gradients_1/pi/dense/MatMul_grad/tuple/control_dependency_1Reshape_4/shape*
Tshape0*
_output_shapes

:Đ*
T0
b
Reshape_5/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_5Reshape<gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_5/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_6/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_6Reshape=gradients_1/pi/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_6/shape*
T0*
Tshape0*
_output_shapes

:
b
Reshape_7/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_7Reshape>gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_7/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_8/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙

	Reshape_8Reshape=gradients_1/pi/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_8/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_9/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_9Reshape>gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_9/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_10/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
n

Reshape_10Reshapegradients_1/AddN_2Reshape_10/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_11/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y

Reshape_11Reshapegradients_1/Softplus_grad/mulReshape_11/shape*
T0*
Tshape0*
_output_shapes
:
O
concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
š
concat_2ConcatV2	Reshape_4	Reshape_5	Reshape_6	Reshape_7	Reshape_8	Reshape_9
Reshape_10
Reshape_11concat_2/axis*

Tidx0*
T0*
N*
_output_shapes

:ě
l
PyFunc_2PyFuncconcat_2*
_output_shapes

:ě*
Tin
2*
Tout
2*
token
pyfunc_2
p
Const_6Const*5
value,B*"  h                       *
dtype0*
_output_shapes
:
S
split_2/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ş
split_2SplitVPyFunc_2Const_6split_2/split_dim*
T0*K
_output_shapes9
7:Đ:::::::*
	num_split*

Tlen0
a
Reshape_12/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
h

Reshape_12Reshapesplit_2Reshape_12/shape*
T0*
Tshape0*
_output_shapes
:	h
[
Reshape_13/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_13Reshape	split_2:1Reshape_13/shape*
T0*
Tshape0*
_output_shapes	
:
a
Reshape_14/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
k

Reshape_14Reshape	split_2:2Reshape_14/shape*
T0*
Tshape0* 
_output_shapes
:

[
Reshape_15/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_15Reshape	split_2:3Reshape_15/shape*
Tshape0*
_output_shapes	
:*
T0
a
Reshape_16/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
j

Reshape_16Reshape	split_2:4Reshape_16/shape*
T0*
Tshape0*
_output_shapes
:	
Z
Reshape_17/shapeConst*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_17Reshape	split_2:5Reshape_17/shape*
T0*
Tshape0*
_output_shapes
:
Z
Reshape_18/shapeConst*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_18Reshape	split_2:6Reshape_18/shape*
_output_shapes
:*
T0*
Tshape0
S
Reshape_19/shapeConst*
valueB *
dtype0*
_output_shapes
: 
a

Reshape_19Reshape	split_2:7Reshape_19/shape*
_output_shapes
: *
T0*
Tshape0

beta1_power_1/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*(
_class
loc:@penalty/penalty_param*
dtype0

beta1_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@penalty/penalty_param*
	container *
shape: 
ž
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
x
beta1_power_1/readIdentitybeta1_power_1*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0

beta2_power_1/initial_valueConst*
valueB
 *wž?*(
_class
loc:@penalty/penalty_param*
dtype0*
_output_shapes
: 

beta2_power_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@penalty/penalty_param
ž
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
x
beta2_power_1/readIdentitybeta2_power_1*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Ť
6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@pi/dense/kernel*
valueB"h      *
dtype0*
_output_shapes
:

,pi/dense/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@pi/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ô
&pi/dense/kernel/Adam/Initializer/zerosFill6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,pi/dense/kernel/Adam/Initializer/zeros/Const*"
_class
loc:@pi/dense/kernel*

index_type0*
_output_shapes
:	h*
T0
Ž
pi/dense/kernel/Adam
VariableV2*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	h*
dtype0*
_output_shapes
:	h
Ú
pi/dense/kernel/Adam/AssignAssignpi/dense/kernel/Adam&pi/dense/kernel/Adam/Initializer/zeros*
_output_shapes
:	h*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(

pi/dense/kernel/Adam/readIdentitypi/dense/kernel/Adam*
_output_shapes
:	h*
T0*"
_class
loc:@pi/dense/kernel
­
8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@pi/dense/kernel*
valueB"h      *
dtype0*
_output_shapes
:

.pi/dense/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@pi/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ú
(pi/dense/kernel/Adam_1/Initializer/zerosFill8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.pi/dense/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	h*
T0*"
_class
loc:@pi/dense/kernel*

index_type0
°
pi/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	h*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	h
ŕ
pi/dense/kernel/Adam_1/AssignAssignpi/dense/kernel/Adam_1(pi/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	h

pi/dense/kernel/Adam_1/readIdentitypi/dense/kernel/Adam_1*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	h

$pi/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
valueB*    
˘
pi/dense/bias/Adam
VariableV2*
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Î
pi/dense/bias/Adam/AssignAssignpi/dense/bias/Adam$pi/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:

pi/dense/bias/Adam/readIdentitypi/dense/bias/Adam*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes	
:

&pi/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
¤
pi/dense/bias/Adam_1
VariableV2*
_output_shapes	
:*
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape:*
dtype0
Ô
pi/dense/bias/Adam_1/AssignAssignpi/dense/bias/Adam_1&pi/dense/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(

pi/dense/bias/Adam_1/readIdentitypi/dense/bias/Adam_1*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes	
:
Ż
8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@pi/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:

.pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ý
(pi/dense_1/kernel/Adam/Initializer/zerosFill8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.pi/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*$
_class
loc:@pi/dense_1/kernel*

index_type0* 
_output_shapes
:

´
pi/dense_1/kernel/Adam
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *$
_class
loc:@pi/dense_1/kernel
ă
pi/dense_1/kernel/Adam/AssignAssignpi/dense_1/kernel/Adam(pi/dense_1/kernel/Adam/Initializer/zeros*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

pi/dense_1/kernel/Adam/readIdentitypi/dense_1/kernel/Adam* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel
ą
:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*$
_class
loc:@pi/dense_1/kernel*
valueB"      *
dtype0

0pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

*pi/dense_1/kernel/Adam_1/Initializer/zerosFill:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0pi/dense_1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel*

index_type0
ś
pi/dense_1/kernel/Adam_1
VariableV2*
shared_name *$
_class
loc:@pi/dense_1/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

é
pi/dense_1/kernel/Adam_1/AssignAssignpi/dense_1/kernel/Adam_1*pi/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:


pi/dense_1/kernel/Adam_1/readIdentitypi/dense_1/kernel/Adam_1* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel

&pi/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ś
pi/dense_1/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@pi/dense_1/bias*
	container *
shape:
Ö
pi/dense_1/bias/Adam/AssignAssignpi/dense_1/bias/Adam&pi/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias

pi/dense_1/bias/Adam/readIdentitypi/dense_1/bias/Adam*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias

(pi/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
valueB*    
¨
pi/dense_1/bias/Adam_1
VariableV2*"
_class
loc:@pi/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ü
pi/dense_1/bias/Adam_1/AssignAssignpi/dense_1/bias/Adam_1(pi/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:

pi/dense_1/bias/Adam_1/readIdentitypi/dense_1/bias/Adam_1*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:
Ż
8pi/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@pi/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:

.pi/dense_2/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@pi/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ü
(pi/dense_2/kernel/Adam/Initializer/zerosFill8pi/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensor.pi/dense_2/kernel/Adam/Initializer/zeros/Const*
T0*$
_class
loc:@pi/dense_2/kernel*

index_type0*
_output_shapes
:	
˛
pi/dense_2/kernel/Adam
VariableV2*
shared_name *$
_class
loc:@pi/dense_2/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
â
pi/dense_2/kernel/Adam/AssignAssignpi/dense_2/kernel/Adam(pi/dense_2/kernel/Adam/Initializer/zeros*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

pi/dense_2/kernel/Adam/readIdentitypi/dense_2/kernel/Adam*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0
ą
:pi/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@pi/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:

0pi/dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *$
_class
loc:@pi/dense_2/kernel*
valueB
 *    

*pi/dense_2/kernel/Adam_1/Initializer/zerosFill:pi/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor0pi/dense_2/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel*

index_type0
´
pi/dense_2/kernel/Adam_1
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *$
_class
loc:@pi/dense_2/kernel*
	container 
č
pi/dense_2/kernel/Adam_1/AssignAssignpi/dense_2/kernel/Adam_1*pi/dense_2/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(

pi/dense_2/kernel/Adam_1/readIdentitypi/dense_2/kernel/Adam_1*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel

&pi/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
¤
pi/dense_2/bias/Adam
VariableV2*"
_class
loc:@pi/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ő
pi/dense_2/bias/Adam/AssignAssignpi/dense_2/bias/Adam&pi/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:

pi/dense_2/bias/Adam/readIdentitypi/dense_2/bias/Adam*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:

(pi/dense_2/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@pi/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
Ś
pi/dense_2/bias/Adam_1
VariableV2*
shared_name *"
_class
loc:@pi/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ű
pi/dense_2/bias/Adam_1/AssignAssignpi/dense_2/bias/Adam_1(pi/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:

pi/dense_2/bias/Adam_1/readIdentitypi/dense_2/bias/Adam_1*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:

!pi/log_std/Adam/Initializer/zerosConst*
_class
loc:@pi/log_std*
valueB*    *
dtype0*
_output_shapes
:

pi/log_std/Adam
VariableV2*
_class
loc:@pi/log_std*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Á
pi/log_std/Adam/AssignAssignpi/log_std/Adam!pi/log_std/Adam/Initializer/zeros*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
u
pi/log_std/Adam/readIdentitypi/log_std/Adam*
_output_shapes
:*
T0*
_class
loc:@pi/log_std

#pi/log_std/Adam_1/Initializer/zerosConst*
_class
loc:@pi/log_std*
valueB*    *
dtype0*
_output_shapes
:

pi/log_std/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@pi/log_std*
	container 
Ç
pi/log_std/Adam_1/AssignAssignpi/log_std/Adam_1#pi/log_std/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(
y
pi/log_std/Adam_1/readIdentitypi/log_std/Adam_1*
T0*
_class
loc:@pi/log_std*
_output_shapes
:

.penalty/penalty_param/Adam_2/Initializer/zerosConst*(
_class
loc:@penalty/penalty_param*
valueB
 *    *
dtype0*
_output_shapes
: 
Ş
penalty/penalty_param/Adam_2
VariableV2*
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@penalty/penalty_param*
	container *
shape: 
ď
#penalty/penalty_param/Adam_2/AssignAssignpenalty/penalty_param/Adam_2.penalty/penalty_param/Adam_2/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 

!penalty/penalty_param/Adam_2/readIdentitypenalty/penalty_param/Adam_2*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 

.penalty/penalty_param/Adam_3/Initializer/zerosConst*(
_class
loc:@penalty/penalty_param*
valueB
 *    *
dtype0*
_output_shapes
: 
Ş
penalty/penalty_param/Adam_3
VariableV2*
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@penalty/penalty_param*
	container *
shape: 
ď
#penalty/penalty_param/Adam_3/AssignAssignpenalty/penalty_param/Adam_3.penalty/penalty_param/Adam_3/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(

!penalty/penalty_param/Adam_3/readIdentitypenalty/penalty_param/Adam_3*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Y
Adam_1/learning_rateConst*
valueB
 *RI9*
dtype0*
_output_shapes
: 
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Ţ
'Adam_1/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_12*
use_locking( *
T0*"
_class
loc:@pi/dense/kernel*
use_nesterov( *
_output_shapes
:	h
Đ
%Adam_1/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_13*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0* 
_class
loc:@pi/dense/bias
é
)Adam_1/update_pi/dense_1/kernel/ApplyAdam	ApplyAdampi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_14*$
_class
loc:@pi/dense_1/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
Ú
'Adam_1/update_pi/dense_1/bias/ApplyAdam	ApplyAdampi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_15*
use_locking( *
T0*"
_class
loc:@pi/dense_1/bias*
use_nesterov( *
_output_shapes	
:
č
)Adam_1/update_pi/dense_2/kernel/ApplyAdam	ApplyAdampi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_16*
T0*$
_class
loc:@pi/dense_2/kernel*
use_nesterov( *
_output_shapes
:	*
use_locking( 
Ů
'Adam_1/update_pi/dense_2/bias/ApplyAdam	ApplyAdampi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_17*
use_locking( *
T0*"
_class
loc:@pi/dense_2/bias*
use_nesterov( *
_output_shapes
:
Ŕ
"Adam_1/update_pi/log_std/ApplyAdam	ApplyAdam
pi/log_stdpi/log_std/Adampi/log_std/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_18*
use_locking( *
T0*
_class
loc:@pi/log_std*
use_nesterov( *
_output_shapes
:
ő
-Adam_1/update_penalty/penalty_param/ApplyAdam	ApplyAdampenalty/penalty_parampenalty/penalty_param/Adam_2penalty/penalty_param/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_19*
use_locking( *
T0*(
_class
loc:@penalty/penalty_param*
use_nesterov( *
_output_shapes
: 
Ń

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1.^Adam_1/update_penalty/penalty_param/ApplyAdam&^Adam_1/update_pi/dense/bias/ApplyAdam(^Adam_1/update_pi/dense/kernel/ApplyAdam(^Adam_1/update_pi/dense_1/bias/ApplyAdam*^Adam_1/update_pi/dense_1/kernel/ApplyAdam(^Adam_1/update_pi/dense_2/bias/ApplyAdam*^Adam_1/update_pi/dense_2/kernel/ApplyAdam#^Adam_1/update_pi/log_std/ApplyAdam*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param
Ś
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
Ó
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2.^Adam_1/update_penalty/penalty_param/ApplyAdam&^Adam_1/update_pi/dense/bias/ApplyAdam(^Adam_1/update_pi/dense/kernel/ApplyAdam(^Adam_1/update_pi/dense_1/bias/ApplyAdam*^Adam_1/update_pi/dense_1/kernel/ApplyAdam(^Adam_1/update_pi/dense_2/bias/ApplyAdam*^Adam_1/update_pi/dense_2/kernel/ApplyAdam#^Adam_1/update_pi/log_std/ApplyAdam*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Ş
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 

Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1.^Adam_1/update_penalty/penalty_param/ApplyAdam&^Adam_1/update_pi/dense/bias/ApplyAdam(^Adam_1/update_pi/dense/kernel/ApplyAdam(^Adam_1/update_pi/dense_1/bias/ApplyAdam*^Adam_1/update_pi/dense_1/kernel/ApplyAdam(^Adam_1/update_pi/dense_2/bias/ApplyAdam*^Adam_1/update_pi/dense_2/kernel/ApplyAdam#^Adam_1/update_pi/log_std/ApplyAdam
l
Reshape_20/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
r

Reshape_20Reshapepi/dense/kernel/readReshape_20/shape*
T0*
Tshape0*
_output_shapes

:Đ
l
Reshape_21/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_21Reshapepi/dense/bias/readReshape_21/shape*
T0*
Tshape0*
_output_shapes	
:
l
Reshape_22/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
t

Reshape_22Reshapepi/dense_1/kernel/readReshape_22/shape*
T0*
Tshape0*
_output_shapes

:
l
Reshape_23/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q

Reshape_23Reshapepi/dense_1/bias/readReshape_23/shape*
_output_shapes	
:*
T0*
Tshape0
l
Reshape_24/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_24Reshapepi/dense_2/kernel/readReshape_24/shape*
_output_shapes	
:*
T0*
Tshape0
l
Reshape_25/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_25Reshapepi/dense_2/bias/readReshape_25/shape*
_output_shapes
:*
T0*
Tshape0
l
Reshape_26/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
k

Reshape_26Reshapepi/log_std/readReshape_26/shape*
_output_shapes
:*
T0*
Tshape0
l
Reshape_27/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
v

Reshape_27Reshapepenalty/penalty_param/readReshape_27/shape*
T0*
Tshape0*
_output_shapes
:
X
concat_3/axisConst^Adam_1*
dtype0*
_output_shapes
: *
value	B : 
ż
concat_3ConcatV2
Reshape_20
Reshape_21
Reshape_22
Reshape_23
Reshape_24
Reshape_25
Reshape_26
Reshape_27concat_3/axis*
N*
_output_shapes

:ě*

Tidx0*
T0
h
PyFunc_3PyFuncconcat_3*
_output_shapes
:*
Tin
2*
Tout
2*
token
pyfunc_3
y
Const_7Const^Adam_1*5
value,B*"  h                       *
dtype0*
_output_shapes
:
\
split_3/split_dimConst^Adam_1*
dtype0*
_output_shapes
: *
value	B : 

split_3SplitVPyFunc_3Const_7split_3/split_dim*
T0*4
_output_shapes"
 ::::::::*
	num_split*

Tlen0
j
Reshape_28/shapeConst^Adam_1*
valueB"h      *
dtype0*
_output_shapes
:
h

Reshape_28Reshapesplit_3Reshape_28/shape*
Tshape0*
_output_shapes
:	h*
T0
d
Reshape_29/shapeConst^Adam_1*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_29Reshape	split_3:1Reshape_29/shape*
T0*
Tshape0*
_output_shapes	
:
j
Reshape_30/shapeConst^Adam_1*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_30Reshape	split_3:2Reshape_30/shape*
Tshape0* 
_output_shapes
:
*
T0
d
Reshape_31/shapeConst^Adam_1*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_31Reshape	split_3:3Reshape_31/shape*
T0*
Tshape0*
_output_shapes	
:
j
Reshape_32/shapeConst^Adam_1*
valueB"      *
dtype0*
_output_shapes
:
j

Reshape_32Reshape	split_3:4Reshape_32/shape*
T0*
Tshape0*
_output_shapes
:	
c
Reshape_33/shapeConst^Adam_1*
_output_shapes
:*
valueB:*
dtype0
e

Reshape_33Reshape	split_3:5Reshape_33/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_34/shapeConst^Adam_1*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_34Reshape	split_3:6Reshape_34/shape*
T0*
Tshape0*
_output_shapes
:
\
Reshape_35/shapeConst^Adam_1*
dtype0*
_output_shapes
: *
valueB 
a

Reshape_35Reshape	split_3:7Reshape_35/shape*
_output_shapes
: *
T0*
Tshape0
Ś
Assign_1Assignpi/dense/kernel
Reshape_28*
_output_shapes
:	h*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(

Assign_2Assignpi/dense/bias
Reshape_29*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Ť
Assign_3Assignpi/dense_1/kernel
Reshape_30*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˘
Assign_4Assignpi/dense_1/bias
Reshape_31*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ş
Assign_5Assignpi/dense_2/kernel
Reshape_32*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ą
Assign_6Assignpi/dense_2/bias
Reshape_33*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(

Assign_7Assign
pi/log_std
Reshape_34*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Š
Assign_8Assignpenalty/penalty_param
Reshape_35*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
u
group_deps_2NoOp^Adam_1	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8
,
group_deps_3NoOp^Adam_1^group_deps_2
U
sub_3SubPlaceholder_4
vf/Squeeze*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
F
powPowsub_3pow/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_8Const*
dtype0*
_output_shapes
:*
valueB: 
Z
Mean_3MeanpowConst_8*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
U
sub_4SubPlaceholder_5
vc/Squeeze*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
pow_1/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
J
pow_1Powsub_4pow_1/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_9Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_4Meanpow_1Const_9*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
=
add_2AddMean_3Mean_4*
T0*
_output_shapes
: 
T
gradients_2/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_2/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_2/FillFillgradients_2/Shapegradients_2/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
B
'gradients_2/add_2_grad/tuple/group_depsNoOp^gradients_2/Fill
˝
/gradients_2/add_2_grad/tuple/control_dependencyIdentitygradients_2/Fill(^gradients_2/add_2_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_2/Fill*
_output_shapes
: 
ż
1gradients_2/add_2_grad/tuple/control_dependency_1Identitygradients_2/Fill(^gradients_2/add_2_grad/tuple/group_deps*
_output_shapes
: *
T0*#
_class
loc:@gradients_2/Fill
o
%gradients_2/Mean_3_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ľ
gradients_2/Mean_3_grad/ReshapeReshape/gradients_2/add_2_grad/tuple/control_dependency%gradients_2/Mean_3_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
`
gradients_2/Mean_3_grad/ShapeShapepow*
T0*
out_type0*
_output_shapes
:
¤
gradients_2/Mean_3_grad/TileTilegradients_2/Mean_3_grad/Reshapegradients_2/Mean_3_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients_2/Mean_3_grad/Shape_1Shapepow*
T0*
out_type0*
_output_shapes
:
b
gradients_2/Mean_3_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_2/Mean_3_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
˘
gradients_2/Mean_3_grad/ProdProdgradients_2/Mean_3_grad/Shape_1gradients_2/Mean_3_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
gradients_2/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ś
gradients_2/Mean_3_grad/Prod_1Prodgradients_2/Mean_3_grad/Shape_2gradients_2/Mean_3_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
c
!gradients_2/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_2/Mean_3_grad/MaximumMaximumgradients_2/Mean_3_grad/Prod_1!gradients_2/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_2/Mean_3_grad/floordivFloorDivgradients_2/Mean_3_grad/Prodgradients_2/Mean_3_grad/Maximum*
T0*
_output_shapes
: 

gradients_2/Mean_3_grad/CastCast gradients_2/Mean_3_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients_2/Mean_3_grad/truedivRealDivgradients_2/Mean_3_grad/Tilegradients_2/Mean_3_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
o
%gradients_2/Mean_4_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ˇ
gradients_2/Mean_4_grad/ReshapeReshape1gradients_2/add_2_grad/tuple/control_dependency_1%gradients_2/Mean_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
b
gradients_2/Mean_4_grad/ShapeShapepow_1*
T0*
out_type0*
_output_shapes
:
¤
gradients_2/Mean_4_grad/TileTilegradients_2/Mean_4_grad/Reshapegradients_2/Mean_4_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
gradients_2/Mean_4_grad/Shape_1Shapepow_1*
_output_shapes
:*
T0*
out_type0
b
gradients_2/Mean_4_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_2/Mean_4_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
˘
gradients_2/Mean_4_grad/ProdProdgradients_2/Mean_4_grad/Shape_1gradients_2/Mean_4_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
gradients_2/Mean_4_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ś
gradients_2/Mean_4_grad/Prod_1Prodgradients_2/Mean_4_grad/Shape_2gradients_2/Mean_4_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
c
!gradients_2/Mean_4_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_2/Mean_4_grad/MaximumMaximumgradients_2/Mean_4_grad/Prod_1!gradients_2/Mean_4_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_2/Mean_4_grad/floordivFloorDivgradients_2/Mean_4_grad/Prodgradients_2/Mean_4_grad/Maximum*
_output_shapes
: *
T0

gradients_2/Mean_4_grad/CastCast gradients_2/Mean_4_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients_2/Mean_4_grad/truedivRealDivgradients_2/Mean_4_grad/Tilegradients_2/Mean_4_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_2/pow_grad/ShapeShapesub_3*
T0*
out_type0*
_output_shapes
:
_
gradients_2/pow_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
ş
*gradients_2/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pow_grad/Shapegradients_2/pow_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
u
gradients_2/pow_grad/mulMulgradients_2/Mean_3_grad/truedivpow/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_2/pow_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
c
gradients_2/pow_grad/subSubpow/ygradients_2/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_2/pow_grad/PowPowsub_3gradients_2/pow_grad/sub*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_2/pow_grad/mul_1Mulgradients_2/pow_grad/mulgradients_2/pow_grad/Pow*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
gradients_2/pow_grad/SumSumgradients_2/pow_grad/mul_1*gradients_2/pow_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients_2/pow_grad/ReshapeReshapegradients_2/pow_grad/Sumgradients_2/pow_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
c
gradients_2/pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients_2/pow_grad/GreaterGreatersub_3gradients_2/pow_grad/Greater/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
i
$gradients_2/pow_grad/ones_like/ShapeShapesub_3*
T0*
out_type0*
_output_shapes
:
i
$gradients_2/pow_grad/ones_like/ConstConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
˛
gradients_2/pow_grad/ones_likeFill$gradients_2/pow_grad/ones_like/Shape$gradients_2/pow_grad/ones_like/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_2/pow_grad/SelectSelectgradients_2/pow_grad/Greatersub_3gradients_2/pow_grad/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
gradients_2/pow_grad/LogLoggradients_2/pow_grad/Select*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
gradients_2/pow_grad/zeros_like	ZerosLikesub_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
gradients_2/pow_grad/Select_1Selectgradients_2/pow_grad/Greatergradients_2/pow_grad/Loggradients_2/pow_grad/zeros_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
gradients_2/pow_grad/mul_2Mulgradients_2/Mean_3_grad/truedivpow*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_2/pow_grad/mul_3Mulgradients_2/pow_grad/mul_2gradients_2/pow_grad/Select_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
gradients_2/pow_grad/Sum_1Sumgradients_2/pow_grad/mul_3,gradients_2/pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients_2/pow_grad/Reshape_1Reshapegradients_2/pow_grad/Sum_1gradients_2/pow_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
m
%gradients_2/pow_grad/tuple/group_depsNoOp^gradients_2/pow_grad/Reshape^gradients_2/pow_grad/Reshape_1
Ţ
-gradients_2/pow_grad/tuple/control_dependencyIdentitygradients_2/pow_grad/Reshape&^gradients_2/pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_2/pow_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
/gradients_2/pow_grad/tuple/control_dependency_1Identitygradients_2/pow_grad/Reshape_1&^gradients_2/pow_grad/tuple/group_deps*1
_class'
%#loc:@gradients_2/pow_grad/Reshape_1*
_output_shapes
: *
T0
a
gradients_2/pow_1_grad/ShapeShapesub_4*
T0*
out_type0*
_output_shapes
:
a
gradients_2/pow_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Ŕ
,gradients_2/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pow_1_grad/Shapegradients_2/pow_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
y
gradients_2/pow_1_grad/mulMulgradients_2/Mean_4_grad/truedivpow_1/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients_2/pow_1_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
i
gradients_2/pow_1_grad/subSubpow_1/ygradients_2/pow_1_grad/sub/y*
T0*
_output_shapes
: 
r
gradients_2/pow_1_grad/PowPowsub_4gradients_2/pow_1_grad/sub*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_2/pow_1_grad/mul_1Mulgradients_2/pow_1_grad/mulgradients_2/pow_1_grad/Pow*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
gradients_2/pow_1_grad/SumSumgradients_2/pow_1_grad/mul_1,gradients_2/pow_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients_2/pow_1_grad/ReshapeReshapegradients_2/pow_1_grad/Sumgradients_2/pow_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
 gradients_2/pow_1_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients_2/pow_1_grad/GreaterGreatersub_4 gradients_2/pow_1_grad/Greater/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
&gradients_2/pow_1_grad/ones_like/ShapeShapesub_4*
_output_shapes
:*
T0*
out_type0
k
&gradients_2/pow_1_grad/ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¸
 gradients_2/pow_1_grad/ones_likeFill&gradients_2/pow_1_grad/ones_like/Shape&gradients_2/pow_1_grad/ones_like/Const*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0

gradients_2/pow_1_grad/SelectSelectgradients_2/pow_1_grad/Greatersub_4 gradients_2/pow_1_grad/ones_like*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
n
gradients_2/pow_1_grad/LogLoggradients_2/pow_1_grad/Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
!gradients_2/pow_1_grad/zeros_like	ZerosLikesub_4*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
gradients_2/pow_1_grad/Select_1Selectgradients_2/pow_1_grad/Greatergradients_2/pow_1_grad/Log!gradients_2/pow_1_grad/zeros_like*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
y
gradients_2/pow_1_grad/mul_2Mulgradients_2/Mean_4_grad/truedivpow_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_2/pow_1_grad/mul_3Mulgradients_2/pow_1_grad/mul_2gradients_2/pow_1_grad/Select_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients_2/pow_1_grad/Sum_1Sumgradients_2/pow_1_grad/mul_3.gradients_2/pow_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

 gradients_2/pow_1_grad/Reshape_1Reshapegradients_2/pow_1_grad/Sum_1gradients_2/pow_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
s
'gradients_2/pow_1_grad/tuple/group_depsNoOp^gradients_2/pow_1_grad/Reshape!^gradients_2/pow_1_grad/Reshape_1
ć
/gradients_2/pow_1_grad/tuple/control_dependencyIdentitygradients_2/pow_1_grad/Reshape(^gradients_2/pow_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_2/pow_1_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
1gradients_2/pow_1_grad/tuple/control_dependency_1Identity gradients_2/pow_1_grad/Reshape_1(^gradients_2/pow_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_2/pow_1_grad/Reshape_1*
_output_shapes
: 
i
gradients_2/sub_3_grad/ShapeShapePlaceholder_4*
_output_shapes
:*
T0*
out_type0
h
gradients_2/sub_3_grad/Shape_1Shape
vf/Squeeze*
out_type0*
_output_shapes
:*
T0
Ŕ
,gradients_2/sub_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/sub_3_grad/Shapegradients_2/sub_3_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ž
gradients_2/sub_3_grad/SumSum-gradients_2/pow_grad/tuple/control_dependency,gradients_2/sub_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients_2/sub_3_grad/ReshapeReshapegradients_2/sub_3_grad/Sumgradients_2/sub_3_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Â
gradients_2/sub_3_grad/Sum_1Sum-gradients_2/pow_grad/tuple/control_dependency.gradients_2/sub_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
b
gradients_2/sub_3_grad/NegNeggradients_2/sub_3_grad/Sum_1*
_output_shapes
:*
T0
Ł
 gradients_2/sub_3_grad/Reshape_1Reshapegradients_2/sub_3_grad/Neggradients_2/sub_3_grad/Shape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
s
'gradients_2/sub_3_grad/tuple/group_depsNoOp^gradients_2/sub_3_grad/Reshape!^gradients_2/sub_3_grad/Reshape_1
ć
/gradients_2/sub_3_grad/tuple/control_dependencyIdentitygradients_2/sub_3_grad/Reshape(^gradients_2/sub_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_2/sub_3_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
1gradients_2/sub_3_grad/tuple/control_dependency_1Identity gradients_2/sub_3_grad/Reshape_1(^gradients_2/sub_3_grad/tuple/group_deps*3
_class)
'%loc:@gradients_2/sub_3_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
i
gradients_2/sub_4_grad/ShapeShapePlaceholder_5*
_output_shapes
:*
T0*
out_type0
h
gradients_2/sub_4_grad/Shape_1Shape
vc/Squeeze*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_2/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/sub_4_grad/Shapegradients_2/sub_4_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ŕ
gradients_2/sub_4_grad/SumSum/gradients_2/pow_1_grad/tuple/control_dependency,gradients_2/sub_4_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_2/sub_4_grad/ReshapeReshapegradients_2/sub_4_grad/Sumgradients_2/sub_4_grad/Shape*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ä
gradients_2/sub_4_grad/Sum_1Sum/gradients_2/pow_1_grad/tuple/control_dependency.gradients_2/sub_4_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
b
gradients_2/sub_4_grad/NegNeggradients_2/sub_4_grad/Sum_1*
_output_shapes
:*
T0
Ł
 gradients_2/sub_4_grad/Reshape_1Reshapegradients_2/sub_4_grad/Neggradients_2/sub_4_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_2/sub_4_grad/tuple/group_depsNoOp^gradients_2/sub_4_grad/Reshape!^gradients_2/sub_4_grad/Reshape_1
ć
/gradients_2/sub_4_grad/tuple/control_dependencyIdentitygradients_2/sub_4_grad/Reshape(^gradients_2/sub_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_2/sub_4_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
1gradients_2/sub_4_grad/tuple/control_dependency_1Identity gradients_2/sub_4_grad/Reshape_1(^gradients_2/sub_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_2/sub_4_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
!gradients_2/vf/Squeeze_grad/ShapeShapevf/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ä
#gradients_2/vf/Squeeze_grad/ReshapeReshape1gradients_2/sub_3_grad/tuple/control_dependency_1!gradients_2/vf/Squeeze_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
!gradients_2/vc/Squeeze_grad/ShapeShapevc/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ä
#gradients_2/vc/Squeeze_grad/ReshapeReshape1gradients_2/sub_4_grad/tuple/control_dependency_1!gradients_2/vc/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_2/vf/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:

4gradients_2/vf/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_2/vf/Squeeze_grad/Reshape0^gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGrad

<gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_2/vf/Squeeze_grad/Reshape5^gradients_2/vf/dense_2/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_2/vf/Squeeze_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_2/vf/dense_2/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

/gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_2/vc/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:

4gradients_2/vc/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_2/vc/Squeeze_grad/Reshape0^gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGrad

<gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_2/vc/Squeeze_grad/Reshape5^gradients_2/vc/dense_2/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_2/vc/Squeeze_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_2/vc/dense_2/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
â
)gradients_2/vf/dense_2/MatMul_grad/MatMulMatMul<gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependencyvf/dense_2/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ô
+gradients_2/vf/dense_2/MatMul_grad/MatMul_1MatMulvf/dense_1/Tanh<gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

3gradients_2/vf/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vf/dense_2/MatMul_grad/MatMul,^gradients_2/vf/dense_2/MatMul_grad/MatMul_1

;gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vf/dense_2/MatMul_grad/MatMul4^gradients_2/vf/dense_2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/vf/dense_2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vf/dense_2/MatMul_grad/MatMul_14^gradients_2/vf/dense_2/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_2/vf/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
â
)gradients_2/vc/dense_2/MatMul_grad/MatMulMatMul<gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependencyvc/dense_2/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ô
+gradients_2/vc/dense_2/MatMul_grad/MatMul_1MatMulvc/dense_1/Tanh<gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

3gradients_2/vc/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vc/dense_2/MatMul_grad/MatMul,^gradients_2/vc/dense_2/MatMul_grad/MatMul_1

;gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vc/dense_2/MatMul_grad/MatMul4^gradients_2/vc/dense_2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/vc/dense_2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vc/dense_2/MatMul_grad/MatMul_14^gradients_2/vc/dense_2/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_2/vc/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	
ś
)gradients_2/vf/dense_1/Tanh_grad/TanhGradTanhGradvf/dense_1/Tanh;gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
)gradients_2/vc/dense_1/Tanh_grad/TanhGradTanhGradvc/dense_1/Tanh;gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
/gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_2/vf/dense_1/Tanh_grad/TanhGrad*
_output_shapes	
:*
T0*
data_formatNHWC

4gradients_2/vf/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_2/vf/dense_1/Tanh_grad/TanhGrad

<gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_2/vf/dense_1/Tanh_grad/TanhGrad5^gradients_2/vf/dense_1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/vf/dense_1/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_2/vf/dense_1/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ś
/gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_2/vc/dense_1/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:*
T0

4gradients_2/vc/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_2/vc/dense_1/Tanh_grad/TanhGrad

<gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_2/vc/dense_1/Tanh_grad/TanhGrad5^gradients_2/vc/dense_1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/vc/dense_1/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_2/vc/dense_1/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
â
)gradients_2/vf/dense_1/MatMul_grad/MatMulMatMul<gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependencyvf/dense_1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ó
+gradients_2/vf/dense_1/MatMul_grad/MatMul_1MatMulvf/dense/Tanh<gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

3gradients_2/vf/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vf/dense_1/MatMul_grad/MatMul,^gradients_2/vf/dense_1/MatMul_grad/MatMul_1

;gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vf/dense_1/MatMul_grad/MatMul4^gradients_2/vf/dense_1/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/vf/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vf/dense_1/MatMul_grad/MatMul_14^gradients_2/vf/dense_1/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_2/vf/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:

â
)gradients_2/vc/dense_1/MatMul_grad/MatMulMatMul<gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependencyvc/dense_1/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ó
+gradients_2/vc/dense_1/MatMul_grad/MatMul_1MatMulvc/dense/Tanh<gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

3gradients_2/vc/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vc/dense_1/MatMul_grad/MatMul,^gradients_2/vc/dense_1/MatMul_grad/MatMul_1

;gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vc/dense_1/MatMul_grad/MatMul4^gradients_2/vc/dense_1/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients_2/vc/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

=gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vc/dense_1/MatMul_grad/MatMul_14^gradients_2/vc/dense_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*>
_class4
20loc:@gradients_2/vc/dense_1/MatMul_grad/MatMul_1
˛
'gradients_2/vf/dense/Tanh_grad/TanhGradTanhGradvf/dense/Tanh;gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˛
'gradients_2/vc/dense/Tanh_grad/TanhGradTanhGradvc/dense/Tanh;gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˘
-gradients_2/vf/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_2/vf/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:*
T0

2gradients_2/vf/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_2/vf/dense/BiasAdd_grad/BiasAddGrad(^gradients_2/vf/dense/Tanh_grad/TanhGrad

:gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_2/vf/dense/Tanh_grad/TanhGrad3^gradients_2/vf/dense/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_2/vf/dense/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_2/vf/dense/BiasAdd_grad/BiasAddGrad3^gradients_2/vf/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*@
_class6
42loc:@gradients_2/vf/dense/BiasAdd_grad/BiasAddGrad
˘
-gradients_2/vc/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_2/vc/dense/Tanh_grad/TanhGrad*
_output_shapes	
:*
T0*
data_formatNHWC

2gradients_2/vc/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_2/vc/dense/BiasAdd_grad/BiasAddGrad(^gradients_2/vc/dense/Tanh_grad/TanhGrad

:gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_2/vc/dense/Tanh_grad/TanhGrad3^gradients_2/vc/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*:
_class0
.,loc:@gradients_2/vc/dense/Tanh_grad/TanhGrad

<gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_2/vc/dense/BiasAdd_grad/BiasAddGrad3^gradients_2/vc/dense/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients_2/vc/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
Ű
'gradients_2/vf/dense/MatMul_grad/MatMulMatMul:gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependencyvf/dense/kernel/read*
transpose_b(*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h*
transpose_a( 
Ě
)gradients_2/vf/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	h*
transpose_a(*
transpose_b( *
T0

1gradients_2/vf/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_2/vf/dense/MatMul_grad/MatMul*^gradients_2/vf/dense/MatMul_grad/MatMul_1

9gradients_2/vf/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_2/vf/dense/MatMul_grad/MatMul2^gradients_2/vf/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_2/vf/dense/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h

;gradients_2/vf/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_2/vf/dense/MatMul_grad/MatMul_12^gradients_2/vf/dense/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/vf/dense/MatMul_grad/MatMul_1*
_output_shapes
:	h
Ű
'gradients_2/vc/dense/MatMul_grad/MatMulMatMul:gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependencyvc/dense/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h*
transpose_a( *
transpose_b(
Ě
)gradients_2/vc/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	h*
transpose_a(*
transpose_b( 

1gradients_2/vc/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_2/vc/dense/MatMul_grad/MatMul*^gradients_2/vc/dense/MatMul_grad/MatMul_1

9gradients_2/vc/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_2/vc/dense/MatMul_grad/MatMul2^gradients_2/vc/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_2/vc/dense/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h

;gradients_2/vc/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_2/vc/dense/MatMul_grad/MatMul_12^gradients_2/vc/dense/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/vc/dense/MatMul_grad/MatMul_1*
_output_shapes
:	h
c
Reshape_36/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_36Reshape;gradients_2/vf/dense/MatMul_grad/tuple/control_dependency_1Reshape_36/shape*
T0*
Tshape0*
_output_shapes

:Đ
c
Reshape_37/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_37Reshape<gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_37/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_38/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_38Reshape=gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_38/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_39/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_39Reshape>gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_39/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_40/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0


Reshape_40Reshape=gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_40/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_41/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_41Reshape>gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_41/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_42/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_42Reshape;gradients_2/vc/dense/MatMul_grad/tuple/control_dependency_1Reshape_42/shape*
T0*
Tshape0*
_output_shapes

:Đ
c
Reshape_43/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_43Reshape<gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_43/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_44/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_44Reshape=gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_44/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_45/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_45Reshape>gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_45/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_46/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_46Reshape=gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_46/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_47/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0


Reshape_47Reshape>gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_47/shape*
T0*
Tshape0*
_output_shapes
:
O
concat_4/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ď
concat_4ConcatV2
Reshape_36
Reshape_37
Reshape_38
Reshape_39
Reshape_40
Reshape_41
Reshape_42
Reshape_43
Reshape_44
Reshape_45
Reshape_46
Reshape_47concat_4/axis*

Tidx0*
T0*
N*
_output_shapes

:Ź
l
PyFunc_4PyFuncconcat_4*
_output_shapes

:Ź*
Tin
2*
Tout
2*
token
pyfunc_4

Const_10Const*E
value<B:"0 h                  h                 *
dtype0*
_output_shapes
:
S
split_4/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ę
split_4SplitVPyFunc_4Const_10split_4/split_dim*
T0*j
_output_shapesX
V:Đ::::::Đ:::::*
	num_split*

Tlen0
a
Reshape_48/shapeConst*
_output_shapes
:*
valueB"h      *
dtype0
h

Reshape_48Reshapesplit_4Reshape_48/shape*
_output_shapes
:	h*
T0*
Tshape0
[
Reshape_49/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_49Reshape	split_4:1Reshape_49/shape*
_output_shapes	
:*
T0*
Tshape0
a
Reshape_50/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_50Reshape	split_4:2Reshape_50/shape*
T0*
Tshape0* 
_output_shapes
:

[
Reshape_51/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_51Reshape	split_4:3Reshape_51/shape*
T0*
Tshape0*
_output_shapes	
:
a
Reshape_52/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
j

Reshape_52Reshape	split_4:4Reshape_52/shape*
T0*
Tshape0*
_output_shapes
:	
Z
Reshape_53/shapeConst*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_53Reshape	split_4:5Reshape_53/shape*
_output_shapes
:*
T0*
Tshape0
a
Reshape_54/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
j

Reshape_54Reshape	split_4:6Reshape_54/shape*
_output_shapes
:	h*
T0*
Tshape0
[
Reshape_55/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_55Reshape	split_4:7Reshape_55/shape*
T0*
Tshape0*
_output_shapes	
:
a
Reshape_56/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_56Reshape	split_4:8Reshape_56/shape* 
_output_shapes
:
*
T0*
Tshape0
[
Reshape_57/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_57Reshape	split_4:9Reshape_57/shape*
_output_shapes	
:*
T0*
Tshape0
a
Reshape_58/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_58Reshape
split_4:10Reshape_58/shape*
T0*
Tshape0*
_output_shapes
:	
Z
Reshape_59/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_59Reshape
split_4:11Reshape_59/shape*
T0*
Tshape0*
_output_shapes
:

beta1_power_2/initial_valueConst*
valueB
 *fff?* 
_class
loc:@vc/dense/bias*
dtype0*
_output_shapes
: 

beta1_power_2
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name * 
_class
loc:@vc/dense/bias*
	container 
ś
beta1_power_2/AssignAssignbeta1_power_2beta1_power_2/initial_value*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
p
beta1_power_2/readIdentitybeta1_power_2*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: 

beta2_power_2/initial_valueConst*
valueB
 *wž?* 
_class
loc:@vc/dense/bias*
dtype0*
_output_shapes
: 

beta2_power_2
VariableV2* 
_class
loc:@vc/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
ś
beta2_power_2/AssignAssignbeta2_power_2beta2_power_2/initial_value*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: 
p
beta2_power_2/readIdentitybeta2_power_2*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
Ť
6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@vf/dense/kernel*
valueB"h      *
dtype0*
_output_shapes
:

,vf/dense/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@vf/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ô
&vf/dense/kernel/Adam/Initializer/zerosFill6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vf/dense/kernel/Adam/Initializer/zeros/Const*
T0*"
_class
loc:@vf/dense/kernel*

index_type0*
_output_shapes
:	h
Ž
vf/dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	h*
shared_name *"
_class
loc:@vf/dense/kernel*
	container *
shape:	h
Ú
vf/dense/kernel/Adam/AssignAssignvf/dense/kernel/Adam&vf/dense/kernel/Adam/Initializer/zeros*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	h*
use_locking(*
T0

vf/dense/kernel/Adam/readIdentityvf/dense/kernel/Adam*
_output_shapes
:	h*
T0*"
_class
loc:@vf/dense/kernel
­
8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@vf/dense/kernel*
valueB"h      *
dtype0*
_output_shapes
:

.vf/dense/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@vf/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ú
(vf/dense/kernel/Adam_1/Initializer/zerosFill8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vf/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@vf/dense/kernel*

index_type0*
_output_shapes
:	h
°
vf/dense/kernel/Adam_1
VariableV2*"
_class
loc:@vf/dense/kernel*
	container *
shape:	h*
dtype0*
_output_shapes
:	h*
shared_name 
ŕ
vf/dense/kernel/Adam_1/AssignAssignvf/dense/kernel/Adam_1(vf/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	h

vf/dense/kernel/Adam_1/readIdentityvf/dense/kernel/Adam_1*
_output_shapes
:	h*
T0*"
_class
loc:@vf/dense/kernel

$vf/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
valueB*    
˘
vf/dense/bias/Adam
VariableV2*
_output_shapes	
:*
shared_name * 
_class
loc:@vf/dense/bias*
	container *
shape:*
dtype0
Î
vf/dense/bias/Adam/AssignAssignvf/dense/bias/Adam$vf/dense/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(

vf/dense/bias/Adam/readIdentityvf/dense/bias/Adam*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias

&vf/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@vf/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
¤
vf/dense/bias/Adam_1
VariableV2*
_output_shapes	
:*
shared_name * 
_class
loc:@vf/dense/bias*
	container *
shape:*
dtype0
Ô
vf/dense/bias/Adam_1/AssignAssignvf/dense/bias/Adam_1&vf/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:

vf/dense/bias/Adam_1/readIdentityvf/dense/bias/Adam_1*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias
Ż
8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*$
_class
loc:@vf/dense_1/kernel*
valueB"      

.vf/dense_1/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@vf/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ý
(vf/dense_1/kernel/Adam/Initializer/zerosFill8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vf/dense_1/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel*

index_type0
´
vf/dense_1/kernel/Adam
VariableV2*
shared_name *$
_class
loc:@vf/dense_1/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ă
vf/dense_1/kernel/Adam/AssignAssignvf/dense_1/kernel/Adam(vf/dense_1/kernel/Adam/Initializer/zeros*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

vf/dense_1/kernel/Adam/readIdentityvf/dense_1/kernel/Adam*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

ą
:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vf/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:

0vf/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *$
_class
loc:@vf/dense_1/kernel*
valueB
 *    *
dtype0

*vf/dense_1/kernel/Adam_1/Initializer/zerosFill:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vf/dense_1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel*

index_type0
ś
vf/dense_1/kernel/Adam_1
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *$
_class
loc:@vf/dense_1/kernel
é
vf/dense_1/kernel/Adam_1/AssignAssignvf/dense_1/kernel/Adam_1*vf/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:


vf/dense_1/kernel/Adam_1/readIdentityvf/dense_1/kernel/Adam_1*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0

&vf/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@vf/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ś
vf/dense_1/bias/Adam
VariableV2*
shared_name *"
_class
loc:@vf/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ö
vf/dense_1/bias/Adam/AssignAssignvf/dense_1/bias/Adam&vf/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias

vf/dense_1/bias/Adam/readIdentityvf/dense_1/bias/Adam*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias

(vf/dense_1/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vf/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
¨
vf/dense_1/bias/Adam_1
VariableV2*"
_class
loc:@vf/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ü
vf/dense_1/bias/Adam_1/AssignAssignvf/dense_1/bias/Adam_1(vf/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:

vf/dense_1/bias/Adam_1/readIdentityvf/dense_1/bias/Adam_1*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
Ľ
(vf/dense_2/kernel/Adam/Initializer/zerosConst*$
_class
loc:@vf/dense_2/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
˛
vf/dense_2/kernel/Adam
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *$
_class
loc:@vf/dense_2/kernel*
	container 
â
vf/dense_2/kernel/Adam/AssignAssignvf/dense_2/kernel/Adam(vf/dense_2/kernel/Adam/Initializer/zeros*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

vf/dense_2/kernel/Adam/readIdentityvf/dense_2/kernel/Adam*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0
§
*vf/dense_2/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@vf/dense_2/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
´
vf/dense_2/kernel/Adam_1
VariableV2*
shared_name *$
_class
loc:@vf/dense_2/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
č
vf/dense_2/kernel/Adam_1/AssignAssignvf/dense_2/kernel/Adam_1*vf/dense_2/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(

vf/dense_2/kernel/Adam_1/readIdentityvf/dense_2/kernel/Adam_1*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	

&vf/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@vf/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
¤
vf/dense_2/bias/Adam
VariableV2*
shared_name *"
_class
loc:@vf/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ő
vf/dense_2/bias/Adam/AssignAssignvf/dense_2/bias/Adam&vf/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias

vf/dense_2/bias/Adam/readIdentityvf/dense_2/bias/Adam*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0

(vf/dense_2/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vf/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
Ś
vf/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@vf/dense_2/bias*
	container *
shape:
Ű
vf/dense_2/bias/Adam_1/AssignAssignvf/dense_2/bias/Adam_1(vf/dense_2/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(

vf/dense_2/bias/Adam_1/readIdentityvf/dense_2/bias/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias
Ť
6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@vc/dense/kernel*
valueB"h      *
dtype0*
_output_shapes
:

,vc/dense/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@vc/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ô
&vc/dense/kernel/Adam/Initializer/zerosFill6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vc/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	h*
T0*"
_class
loc:@vc/dense/kernel*

index_type0
Ž
vc/dense/kernel/Adam
VariableV2*
_output_shapes
:	h*
shared_name *"
_class
loc:@vc/dense/kernel*
	container *
shape:	h*
dtype0
Ú
vc/dense/kernel/Adam/AssignAssignvc/dense/kernel/Adam&vc/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	h

vc/dense/kernel/Adam/readIdentityvc/dense/kernel/Adam*
_output_shapes
:	h*
T0*"
_class
loc:@vc/dense/kernel
­
8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@vc/dense/kernel*
valueB"h      *
dtype0*
_output_shapes
:

.vc/dense/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@vc/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ú
(vc/dense/kernel/Adam_1/Initializer/zerosFill8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vc/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@vc/dense/kernel*

index_type0*
_output_shapes
:	h
°
vc/dense/kernel/Adam_1
VariableV2*
	container *
shape:	h*
dtype0*
_output_shapes
:	h*
shared_name *"
_class
loc:@vc/dense/kernel
ŕ
vc/dense/kernel/Adam_1/AssignAssignvc/dense/kernel/Adam_1(vc/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	h

vc/dense/kernel/Adam_1/readIdentityvc/dense/kernel/Adam_1*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	h

$vc/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@vc/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
˘
vc/dense/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name * 
_class
loc:@vc/dense/bias
Î
vc/dense/bias/Adam/AssignAssignvc/dense/bias/Adam$vc/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:

vc/dense/bias/Adam/readIdentityvc/dense/bias/Adam*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias

&vc/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@vc/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
¤
vc/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name * 
_class
loc:@vc/dense/bias*
	container *
shape:
Ô
vc/dense/bias/Adam_1/AssignAssignvc/dense/bias/Adam_1&vc/dense/bias/Adam_1/Initializer/zeros* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

vc/dense/bias/Adam_1/readIdentityvc/dense/bias/Adam_1*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias
Ż
8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vc/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:

.vc/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *$
_class
loc:@vc/dense_1/kernel*
valueB
 *    *
dtype0
ý
(vc/dense_1/kernel/Adam/Initializer/zerosFill8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vc/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*$
_class
loc:@vc/dense_1/kernel*

index_type0* 
_output_shapes
:

´
vc/dense_1/kernel/Adam
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *$
_class
loc:@vc/dense_1/kernel
ă
vc/dense_1/kernel/Adam/AssignAssignvc/dense_1/kernel/Adam(vc/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:


vc/dense_1/kernel/Adam/readIdentityvc/dense_1/kernel/Adam*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

ą
:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*$
_class
loc:@vc/dense_1/kernel*
valueB"      *
dtype0

0vc/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@vc/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

*vc/dense_1/kernel/Adam_1/Initializer/zerosFill:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vc/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*$
_class
loc:@vc/dense_1/kernel*

index_type0* 
_output_shapes
:

ś
vc/dense_1/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *$
_class
loc:@vc/dense_1/kernel*
	container *
shape:

é
vc/dense_1/kernel/Adam_1/AssignAssignvc/dense_1/kernel/Adam_1*vc/dense_1/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(

vc/dense_1/kernel/Adam_1/readIdentityvc/dense_1/kernel/Adam_1* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel

&vc/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@vc/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ś
vc/dense_1/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@vc/dense_1/bias*
	container *
shape:
Ö
vc/dense_1/bias/Adam/AssignAssignvc/dense_1/bias/Adam&vc/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:

vc/dense_1/bias/Adam/readIdentityvc/dense_1/bias/Adam*
_output_shapes	
:*
T0*"
_class
loc:@vc/dense_1/bias

(vc/dense_1/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vc/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
¨
vc/dense_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@vc/dense_1/bias*
	container *
shape:
Ü
vc/dense_1/bias/Adam_1/AssignAssignvc/dense_1/bias/Adam_1(vc/dense_1/bias/Adam_1/Initializer/zeros*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

vc/dense_1/bias/Adam_1/readIdentityvc/dense_1/bias/Adam_1*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
Ľ
(vc/dense_2/kernel/Adam/Initializer/zerosConst*$
_class
loc:@vc/dense_2/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
˛
vc/dense_2/kernel/Adam
VariableV2*$
_class
loc:@vc/dense_2/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
â
vc/dense_2/kernel/Adam/AssignAssignvc/dense_2/kernel/Adam(vc/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	

vc/dense_2/kernel/Adam/readIdentityvc/dense_2/kernel/Adam*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	
§
*vc/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
valueB	*    
´
vc/dense_2/kernel/Adam_1
VariableV2*
shared_name *$
_class
loc:@vc/dense_2/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
č
vc/dense_2/kernel/Adam_1/AssignAssignvc/dense_2/kernel/Adam_1*vc/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	

vc/dense_2/kernel/Adam_1/readIdentityvc/dense_2/kernel/Adam_1*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel

&vc/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@vc/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
¤
vc/dense_2/bias/Adam
VariableV2*
_output_shapes
:*
shared_name *"
_class
loc:@vc/dense_2/bias*
	container *
shape:*
dtype0
Ő
vc/dense_2/bias/Adam/AssignAssignvc/dense_2/bias/Adam&vc/dense_2/bias/Adam/Initializer/zeros*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

vc/dense_2/bias/Adam/readIdentityvc/dense_2/bias/Adam*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:

(vc/dense_2/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
valueB*    
Ś
vc/dense_2/bias/Adam_1
VariableV2*
_output_shapes
:*
shared_name *"
_class
loc:@vc/dense_2/bias*
	container *
shape:*
dtype0
Ű
vc/dense_2/bias/Adam_1/AssignAssignvc/dense_2/bias/Adam_1(vc/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:

vc/dense_2/bias/Adam_1/readIdentityvc/dense_2/bias/Adam_1*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
Y
Adam_2/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_2/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
Q
Adam_2/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_2/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Ţ
'Adam_2/update_vf/dense/kernel/ApplyAdam	ApplyAdamvf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_48*"
_class
loc:@vf/dense/kernel*
use_nesterov( *
_output_shapes
:	h*
use_locking( *
T0
Đ
%Adam_2/update_vf/dense/bias/ApplyAdam	ApplyAdamvf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_49*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0* 
_class
loc:@vf/dense/bias
é
)Adam_2/update_vf/dense_1/kernel/ApplyAdam	ApplyAdamvf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_50*
use_locking( *
T0*$
_class
loc:@vf/dense_1/kernel*
use_nesterov( * 
_output_shapes
:

Ú
'Adam_2/update_vf/dense_1/bias/ApplyAdam	ApplyAdamvf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_51*
T0*"
_class
loc:@vf/dense_1/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
č
)Adam_2/update_vf/dense_2/kernel/ApplyAdam	ApplyAdamvf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_52*
T0*$
_class
loc:@vf/dense_2/kernel*
use_nesterov( *
_output_shapes
:	*
use_locking( 
Ů
'Adam_2/update_vf/dense_2/bias/ApplyAdam	ApplyAdamvf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_53*
use_locking( *
T0*"
_class
loc:@vf/dense_2/bias*
use_nesterov( *
_output_shapes
:
Ţ
'Adam_2/update_vc/dense/kernel/ApplyAdam	ApplyAdamvc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_54*
T0*"
_class
loc:@vc/dense/kernel*
use_nesterov( *
_output_shapes
:	h*
use_locking( 
Đ
%Adam_2/update_vc/dense/bias/ApplyAdam	ApplyAdamvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_55*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0* 
_class
loc:@vc/dense/bias
é
)Adam_2/update_vc/dense_1/kernel/ApplyAdam	ApplyAdamvc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_56* 
_output_shapes
:
*
use_locking( *
T0*$
_class
loc:@vc/dense_1/kernel*
use_nesterov( 
Ú
'Adam_2/update_vc/dense_1/bias/ApplyAdam	ApplyAdamvc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_57*
_output_shapes	
:*
use_locking( *
T0*"
_class
loc:@vc/dense_1/bias*
use_nesterov( 
č
)Adam_2/update_vc/dense_2/kernel/ApplyAdam	ApplyAdamvc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_58*
use_locking( *
T0*$
_class
loc:@vc/dense_2/kernel*
use_nesterov( *
_output_shapes
:	
Ů
'Adam_2/update_vc/dense_2/bias/ApplyAdam	ApplyAdamvc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_59*
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@vc/dense_2/bias*
use_nesterov( 
ň

Adam_2/mulMulbeta1_power_2/readAdam_2/beta1&^Adam_2/update_vc/dense/bias/ApplyAdam(^Adam_2/update_vc/dense/kernel/ApplyAdam(^Adam_2/update_vc/dense_1/bias/ApplyAdam*^Adam_2/update_vc/dense_1/kernel/ApplyAdam(^Adam_2/update_vc/dense_2/bias/ApplyAdam*^Adam_2/update_vc/dense_2/kernel/ApplyAdam&^Adam_2/update_vf/dense/bias/ApplyAdam(^Adam_2/update_vf/dense/kernel/ApplyAdam(^Adam_2/update_vf/dense_1/bias/ApplyAdam*^Adam_2/update_vf/dense_1/kernel/ApplyAdam(^Adam_2/update_vf/dense_2/bias/ApplyAdam*^Adam_2/update_vf/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: 

Adam_2/AssignAssignbeta1_power_2
Adam_2/mul* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
ô
Adam_2/mul_1Mulbeta2_power_2/readAdam_2/beta2&^Adam_2/update_vc/dense/bias/ApplyAdam(^Adam_2/update_vc/dense/kernel/ApplyAdam(^Adam_2/update_vc/dense_1/bias/ApplyAdam*^Adam_2/update_vc/dense_1/kernel/ApplyAdam(^Adam_2/update_vc/dense_2/bias/ApplyAdam*^Adam_2/update_vc/dense_2/kernel/ApplyAdam&^Adam_2/update_vf/dense/bias/ApplyAdam(^Adam_2/update_vf/dense/kernel/ApplyAdam(^Adam_2/update_vf/dense_1/bias/ApplyAdam*^Adam_2/update_vf/dense_1/kernel/ApplyAdam(^Adam_2/update_vf/dense_2/bias/ApplyAdam*^Adam_2/update_vf/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias
˘
Adam_2/Assign_1Assignbeta2_power_2Adam_2/mul_1*
use_locking( *
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: 
Ź
Adam_2NoOp^Adam_2/Assign^Adam_2/Assign_1&^Adam_2/update_vc/dense/bias/ApplyAdam(^Adam_2/update_vc/dense/kernel/ApplyAdam(^Adam_2/update_vc/dense_1/bias/ApplyAdam*^Adam_2/update_vc/dense_1/kernel/ApplyAdam(^Adam_2/update_vc/dense_2/bias/ApplyAdam*^Adam_2/update_vc/dense_2/kernel/ApplyAdam&^Adam_2/update_vf/dense/bias/ApplyAdam(^Adam_2/update_vf/dense/kernel/ApplyAdam(^Adam_2/update_vf/dense_1/bias/ApplyAdam*^Adam_2/update_vf/dense_1/kernel/ApplyAdam(^Adam_2/update_vf/dense_2/bias/ApplyAdam*^Adam_2/update_vf/dense_2/kernel/ApplyAdam
l
Reshape_60/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
r

Reshape_60Reshapevf/dense/kernel/readReshape_60/shape*
_output_shapes

:Đ*
T0*
Tshape0
l
Reshape_61/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_61Reshapevf/dense/bias/readReshape_61/shape*
Tshape0*
_output_shapes	
:*
T0
l
Reshape_62/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t

Reshape_62Reshapevf/dense_1/kernel/readReshape_62/shape*
_output_shapes

:*
T0*
Tshape0
l
Reshape_63/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_63Reshapevf/dense_1/bias/readReshape_63/shape*
_output_shapes	
:*
T0*
Tshape0
l
Reshape_64/shapeConst^Adam_2*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
s

Reshape_64Reshapevf/dense_2/kernel/readReshape_64/shape*
Tshape0*
_output_shapes	
:*
T0
l
Reshape_65/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
p

Reshape_65Reshapevf/dense_2/bias/readReshape_65/shape*
T0*
Tshape0*
_output_shapes
:
l
Reshape_66/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
r

Reshape_66Reshapevc/dense/kernel/readReshape_66/shape*
Tshape0*
_output_shapes

:Đ*
T0
l
Reshape_67/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
o

Reshape_67Reshapevc/dense/bias/readReshape_67/shape*
T0*
Tshape0*
_output_shapes	
:
l
Reshape_68/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t

Reshape_68Reshapevc/dense_1/kernel/readReshape_68/shape*
T0*
Tshape0*
_output_shapes

:
l
Reshape_69/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q

Reshape_69Reshapevc/dense_1/bias/readReshape_69/shape*
_output_shapes	
:*
T0*
Tshape0
l
Reshape_70/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_70Reshapevc/dense_2/kernel/readReshape_70/shape*
T0*
Tshape0*
_output_shapes	
:
l
Reshape_71/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_71Reshapevc/dense_2/bias/readReshape_71/shape*
T0*
Tshape0*
_output_shapes
:
X
concat_5/axisConst^Adam_2*
value	B : *
dtype0*
_output_shapes
: 
ď
concat_5ConcatV2
Reshape_60
Reshape_61
Reshape_62
Reshape_63
Reshape_64
Reshape_65
Reshape_66
Reshape_67
Reshape_68
Reshape_69
Reshape_70
Reshape_71concat_5/axis*
T0*
N*
_output_shapes

:Ź*

Tidx0
h
PyFunc_5PyFuncconcat_5*
Tout
2*
token
pyfunc_5*
_output_shapes
:*
Tin
2

Const_11Const^Adam_2*E
value<B:"0 h                  h                 *
dtype0*
_output_shapes
:
\
split_5/split_dimConst^Adam_2*
value	B : *
dtype0*
_output_shapes
: 
¤
split_5SplitVPyFunc_5Const_11split_5/split_dim*
T0*D
_output_shapes2
0::::::::::::*
	num_split*

Tlen0
j
Reshape_72/shapeConst^Adam_2*
valueB"h      *
dtype0*
_output_shapes
:
h

Reshape_72Reshapesplit_5Reshape_72/shape*
T0*
Tshape0*
_output_shapes
:	h
d
Reshape_73/shapeConst^Adam_2*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_73Reshape	split_5:1Reshape_73/shape*
T0*
Tshape0*
_output_shapes	
:
j
Reshape_74/shapeConst^Adam_2*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_74Reshape	split_5:2Reshape_74/shape* 
_output_shapes
:
*
T0*
Tshape0
d
Reshape_75/shapeConst^Adam_2*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_75Reshape	split_5:3Reshape_75/shape*
T0*
Tshape0*
_output_shapes	
:
j
Reshape_76/shapeConst^Adam_2*
valueB"      *
dtype0*
_output_shapes
:
j

Reshape_76Reshape	split_5:4Reshape_76/shape*
T0*
Tshape0*
_output_shapes
:	
c
Reshape_77/shapeConst^Adam_2*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_77Reshape	split_5:5Reshape_77/shape*
T0*
Tshape0*
_output_shapes
:
j
Reshape_78/shapeConst^Adam_2*
valueB"h      *
dtype0*
_output_shapes
:
j

Reshape_78Reshape	split_5:6Reshape_78/shape*
_output_shapes
:	h*
T0*
Tshape0
d
Reshape_79/shapeConst^Adam_2*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_79Reshape	split_5:7Reshape_79/shape*
_output_shapes	
:*
T0*
Tshape0
j
Reshape_80/shapeConst^Adam_2*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_80Reshape	split_5:8Reshape_80/shape*
T0*
Tshape0* 
_output_shapes
:

d
Reshape_81/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_81Reshape	split_5:9Reshape_81/shape*
Tshape0*
_output_shapes	
:*
T0
j
Reshape_82/shapeConst^Adam_2*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_82Reshape
split_5:10Reshape_82/shape*
T0*
Tshape0*
_output_shapes
:	
c
Reshape_83/shapeConst^Adam_2*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_83Reshape
split_5:11Reshape_83/shape*
_output_shapes
:*
T0*
Tshape0
Ś
Assign_9Assignvf/dense/kernel
Reshape_72*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	h

	Assign_10Assignvf/dense/bias
Reshape_73*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:
Ź
	Assign_11Assignvf/dense_1/kernel
Reshape_74*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ł
	Assign_12Assignvf/dense_1/bias
Reshape_75*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias
Ť
	Assign_13Assignvf/dense_2/kernel
Reshape_76*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
˘
	Assign_14Assignvf/dense_2/bias
Reshape_77*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(
§
	Assign_15Assignvc/dense/kernel
Reshape_78*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	h

	Assign_16Assignvc/dense/bias
Reshape_79*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(
Ź
	Assign_17Assignvc/dense_1/kernel
Reshape_80*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ł
	Assign_18Assignvc/dense_1/bias
Reshape_81*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:
Ť
	Assign_19Assignvc/dense_2/kernel
Reshape_82*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
˘
	Assign_20Assignvc/dense_2/bias
Reshape_83*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:
Ź
group_deps_4NoOp^Adam_2
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
^Assign_20	^Assign_9
,
group_deps_5NoOp^Adam_2^group_deps_4
ż
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta1_power_2/Assign^beta2_power/Assign^beta2_power_1/Assign^beta2_power_2/Assign"^penalty/penalty_param/Adam/Assign$^penalty/penalty_param/Adam_1/Assign$^penalty/penalty_param/Adam_2/Assign$^penalty/penalty_param/Adam_3/Assign^penalty/penalty_param/Assign^pi/dense/bias/Adam/Assign^pi/dense/bias/Adam_1/Assign^pi/dense/bias/Assign^pi/dense/kernel/Adam/Assign^pi/dense/kernel/Adam_1/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Adam/Assign^pi/dense_1/bias/Adam_1/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Adam/Assign ^pi/dense_1/kernel/Adam_1/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Adam/Assign^pi/dense_2/bias/Adam_1/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Adam/Assign ^pi/dense_2/kernel/Adam_1/Assign^pi/dense_2/kernel/Assign^pi/log_std/Adam/Assign^pi/log_std/Adam_1/Assign^pi/log_std/Assign^vc/dense/bias/Adam/Assign^vc/dense/bias/Adam_1/Assign^vc/dense/bias/Assign^vc/dense/kernel/Adam/Assign^vc/dense/kernel/Adam_1/Assign^vc/dense/kernel/Assign^vc/dense_1/bias/Adam/Assign^vc/dense_1/bias/Adam_1/Assign^vc/dense_1/bias/Assign^vc/dense_1/kernel/Adam/Assign ^vc/dense_1/kernel/Adam_1/Assign^vc/dense_1/kernel/Assign^vc/dense_2/bias/Adam/Assign^vc/dense_2/bias/Adam_1/Assign^vc/dense_2/bias/Assign^vc/dense_2/kernel/Adam/Assign ^vc/dense_2/kernel/Adam_1/Assign^vc/dense_2/kernel/Assign^vf/dense/bias/Adam/Assign^vf/dense/bias/Adam_1/Assign^vf/dense/bias/Assign^vf/dense/kernel/Adam/Assign^vf/dense/kernel/Adam_1/Assign^vf/dense/kernel/Assign^vf/dense_1/bias/Adam/Assign^vf/dense_1/bias/Adam_1/Assign^vf/dense_1/bias/Assign^vf/dense_1/kernel/Adam/Assign ^vf/dense_1/kernel/Adam_1/Assign^vf/dense_1/kernel/Assign^vf/dense_2/bias/Adam/Assign^vf/dense_2/bias/Adam_1/Assign^vf/dense_2/bias/Assign^vf/dense_2/kernel/Adam/Assign ^vf/dense_2/kernel/Adam_1/Assign^vf/dense_2/kernel/Assign
c
Reshape_84/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
r

Reshape_84Reshapepi/dense/kernel/readReshape_84/shape*
T0*
Tshape0*
_output_shapes

:Đ
c
Reshape_85/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_85Reshapepi/dense/bias/readReshape_85/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_86/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t

Reshape_86Reshapepi/dense_1/kernel/readReshape_86/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_87/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q

Reshape_87Reshapepi/dense_1/bias/readReshape_87/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_88/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_88Reshapepi/dense_2/kernel/readReshape_88/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_89/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_89Reshapepi/dense_2/bias/readReshape_89/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_90/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
k

Reshape_90Reshapepi/log_std/readReshape_90/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_91/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
r

Reshape_91Reshapevf/dense/kernel/readReshape_91/shape*
T0*
Tshape0*
_output_shapes

:Đ
c
Reshape_92/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_92Reshapevf/dense/bias/readReshape_92/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_93/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
t

Reshape_93Reshapevf/dense_1/kernel/readReshape_93/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_94/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q

Reshape_94Reshapevf/dense_1/bias/readReshape_94/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_95/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
s

Reshape_95Reshapevf/dense_2/kernel/readReshape_95/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_96/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_96Reshapevf/dense_2/bias/readReshape_96/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_97/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
r

Reshape_97Reshapevc/dense/kernel/readReshape_97/shape*
Tshape0*
_output_shapes

:Đ*
T0
c
Reshape_98/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
o

Reshape_98Reshapevc/dense/bias/readReshape_98/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_99/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t

Reshape_99Reshapevc/dense_1/kernel/readReshape_99/shape*
_output_shapes

:*
T0*
Tshape0
d
Reshape_100/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s
Reshape_100Reshapevc/dense_1/bias/readReshape_100/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_101/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
u
Reshape_101Reshapevc/dense_2/kernel/readReshape_101/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_102/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
r
Reshape_102Reshapevc/dense_2/bias/readReshape_102/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_103/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x
Reshape_103Reshapepenalty/penalty_param/readReshape_103/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_104/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
n
Reshape_104Reshapebeta1_power/readReshape_104/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_105/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
n
Reshape_105Reshapebeta2_power/readReshape_105/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_106/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
}
Reshape_106Reshapepenalty/penalty_param/Adam/readReshape_106/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_107/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

Reshape_107Reshape!penalty/penalty_param/Adam_1/readReshape_107/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_108/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p
Reshape_108Reshapebeta1_power_1/readReshape_108/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_109/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p
Reshape_109Reshapebeta2_power_1/readReshape_109/shape*
Tshape0*
_output_shapes
:*
T0
d
Reshape_110/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y
Reshape_110Reshapepi/dense/kernel/Adam/readReshape_110/shape*
T0*
Tshape0*
_output_shapes

:Đ
d
Reshape_111/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{
Reshape_111Reshapepi/dense/kernel/Adam_1/readReshape_111/shape*
Tshape0*
_output_shapes

:Đ*
T0
d
Reshape_112/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
v
Reshape_112Reshapepi/dense/bias/Adam/readReshape_112/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_113/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x
Reshape_113Reshapepi/dense/bias/Adam_1/readReshape_113/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_114/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{
Reshape_114Reshapepi/dense_1/kernel/Adam/readReshape_114/shape*
T0*
Tshape0*
_output_shapes

:
d
Reshape_115/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
}
Reshape_115Reshapepi/dense_1/kernel/Adam_1/readReshape_115/shape*
_output_shapes

:*
T0*
Tshape0
d
Reshape_116/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x
Reshape_116Reshapepi/dense_1/bias/Adam/readReshape_116/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_117/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
z
Reshape_117Reshapepi/dense_1/bias/Adam_1/readReshape_117/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_118/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_118Reshapepi/dense_2/kernel/Adam/readReshape_118/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_119/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
|
Reshape_119Reshapepi/dense_2/kernel/Adam_1/readReshape_119/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_120/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
w
Reshape_120Reshapepi/dense_2/bias/Adam/readReshape_120/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_121/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y
Reshape_121Reshapepi/dense_2/bias/Adam_1/readReshape_121/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_122/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
r
Reshape_122Reshapepi/log_std/Adam/readReshape_122/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_123/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t
Reshape_123Reshapepi/log_std/Adam_1/readReshape_123/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_124/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

Reshape_124Reshape!penalty/penalty_param/Adam_2/readReshape_124/shape*
Tshape0*
_output_shapes
:*
T0
d
Reshape_125/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

Reshape_125Reshape!penalty/penalty_param/Adam_3/readReshape_125/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_126/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p
Reshape_126Reshapebeta1_power_2/readReshape_126/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_127/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p
Reshape_127Reshapebeta2_power_2/readReshape_127/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_128/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y
Reshape_128Reshapevf/dense/kernel/Adam/readReshape_128/shape*
Tshape0*
_output_shapes

:Đ*
T0
d
Reshape_129/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{
Reshape_129Reshapevf/dense/kernel/Adam_1/readReshape_129/shape*
T0*
Tshape0*
_output_shapes

:Đ
d
Reshape_130/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
v
Reshape_130Reshapevf/dense/bias/Adam/readReshape_130/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_131/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x
Reshape_131Reshapevf/dense/bias/Adam_1/readReshape_131/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_132/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{
Reshape_132Reshapevf/dense_1/kernel/Adam/readReshape_132/shape*
T0*
Tshape0*
_output_shapes

:
d
Reshape_133/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
}
Reshape_133Reshapevf/dense_1/kernel/Adam_1/readReshape_133/shape*
T0*
Tshape0*
_output_shapes

:
d
Reshape_134/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x
Reshape_134Reshapevf/dense_1/bias/Adam/readReshape_134/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_135/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
z
Reshape_135Reshapevf/dense_1/bias/Adam_1/readReshape_135/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_136/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
z
Reshape_136Reshapevf/dense_2/kernel/Adam/readReshape_136/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_137/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
|
Reshape_137Reshapevf/dense_2/kernel/Adam_1/readReshape_137/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_138/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
w
Reshape_138Reshapevf/dense_2/bias/Adam/readReshape_138/shape*
Tshape0*
_output_shapes
:*
T0
d
Reshape_139/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y
Reshape_139Reshapevf/dense_2/bias/Adam_1/readReshape_139/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_140/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y
Reshape_140Reshapevc/dense/kernel/Adam/readReshape_140/shape*
T0*
Tshape0*
_output_shapes

:Đ
d
Reshape_141/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{
Reshape_141Reshapevc/dense/kernel/Adam_1/readReshape_141/shape*
Tshape0*
_output_shapes

:Đ*
T0
d
Reshape_142/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
v
Reshape_142Reshapevc/dense/bias/Adam/readReshape_142/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_143/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
x
Reshape_143Reshapevc/dense/bias/Adam_1/readReshape_143/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_144/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
{
Reshape_144Reshapevc/dense_1/kernel/Adam/readReshape_144/shape*
T0*
Tshape0*
_output_shapes

:
d
Reshape_145/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
}
Reshape_145Reshapevc/dense_1/kernel/Adam_1/readReshape_145/shape*
T0*
Tshape0*
_output_shapes

:
d
Reshape_146/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
x
Reshape_146Reshapevc/dense_1/bias/Adam/readReshape_146/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_147/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
z
Reshape_147Reshapevc/dense_1/bias/Adam_1/readReshape_147/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_148/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
z
Reshape_148Reshapevc/dense_2/kernel/Adam/readReshape_148/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_149/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
|
Reshape_149Reshapevc/dense_2/kernel/Adam_1/readReshape_149/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_150/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
w
Reshape_150Reshapevc/dense_2/bias/Adam/readReshape_150/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_151/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y
Reshape_151Reshapevc/dense_2/bias/Adam_1/readReshape_151/shape*
_output_shapes
:*
T0*
Tshape0
O
concat_6/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ă
concat_6ConcatV2
Reshape_84
Reshape_85
Reshape_86
Reshape_87
Reshape_88
Reshape_89
Reshape_90
Reshape_91
Reshape_92
Reshape_93
Reshape_94
Reshape_95
Reshape_96
Reshape_97
Reshape_98
Reshape_99Reshape_100Reshape_101Reshape_102Reshape_103Reshape_104Reshape_105Reshape_106Reshape_107Reshape_108Reshape_109Reshape_110Reshape_111Reshape_112Reshape_113Reshape_114Reshape_115Reshape_116Reshape_117Reshape_118Reshape_119Reshape_120Reshape_121Reshape_122Reshape_123Reshape_124Reshape_125Reshape_126Reshape_127Reshape_128Reshape_129Reshape_130Reshape_131Reshape_132Reshape_133Reshape_134Reshape_135Reshape_136Reshape_137Reshape_138Reshape_139Reshape_140Reshape_141Reshape_142Reshape_143Reshape_144Reshape_145Reshape_146Reshape_147Reshape_148Reshape_149Reshape_150Reshape_151concat_6/axis*

Tidx0*
T0*
ND*
_output_shapes

:ŮČ3
h
PyFunc_6PyFuncconcat_6*
_output_shapes
:*
Tin
2*
Tout
2*
token
pyfunc_6
ĺ
Const_12Const*¨
valueBD" h                     h                  h                                       h   h                                                   h   h                                 h   h                                *
dtype0*
_output_shapes
:D
S
split_6/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 

split_6SplitVPyFunc_6Const_12split_6/split_dim*
T0*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
	num_splitD*

Tlen0
b
Reshape_152/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
j
Reshape_152Reshapesplit_6Reshape_152/shape*
T0*
Tshape0*
_output_shapes
:	h
\
Reshape_153/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_153Reshape	split_6:1Reshape_153/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_154/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_154Reshape	split_6:2Reshape_154/shape* 
_output_shapes
:
*
T0*
Tshape0
\
Reshape_155/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_155Reshape	split_6:3Reshape_155/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_156/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
l
Reshape_156Reshape	split_6:4Reshape_156/shape*
_output_shapes
:	*
T0*
Tshape0
[
Reshape_157/shapeConst*
valueB:*
dtype0*
_output_shapes
:
g
Reshape_157Reshape	split_6:5Reshape_157/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_158/shapeConst*
valueB:*
dtype0*
_output_shapes
:
g
Reshape_158Reshape	split_6:6Reshape_158/shape*
T0*
Tshape0*
_output_shapes
:
b
Reshape_159/shapeConst*
_output_shapes
:*
valueB"h      *
dtype0
l
Reshape_159Reshape	split_6:7Reshape_159/shape*
_output_shapes
:	h*
T0*
Tshape0
\
Reshape_160/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_160Reshape	split_6:8Reshape_160/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_161/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_161Reshape	split_6:9Reshape_161/shape* 
_output_shapes
:
*
T0*
Tshape0
\
Reshape_162/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_162Reshape
split_6:10Reshape_162/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_163/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_163Reshape
split_6:11Reshape_163/shape*
Tshape0*
_output_shapes
:	*
T0
[
Reshape_164/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_164Reshape
split_6:12Reshape_164/shape*
T0*
Tshape0*
_output_shapes
:
b
Reshape_165/shapeConst*
dtype0*
_output_shapes
:*
valueB"h      
m
Reshape_165Reshape
split_6:13Reshape_165/shape*
Tshape0*
_output_shapes
:	h*
T0
\
Reshape_166/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_166Reshape
split_6:14Reshape_166/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_167/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
n
Reshape_167Reshape
split_6:15Reshape_167/shape* 
_output_shapes
:
*
T0*
Tshape0
\
Reshape_168/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_168Reshape
split_6:16Reshape_168/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_169/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_169Reshape
split_6:17Reshape_169/shape*
_output_shapes
:	*
T0*
Tshape0
[
Reshape_170/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_170Reshape
split_6:18Reshape_170/shape*
T0*
Tshape0*
_output_shapes
:
T
Reshape_171/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_171Reshape
split_6:19Reshape_171/shape*
T0*
Tshape0*
_output_shapes
: 
T
Reshape_172/shapeConst*
dtype0*
_output_shapes
: *
valueB 
d
Reshape_172Reshape
split_6:20Reshape_172/shape*
T0*
Tshape0*
_output_shapes
: 
T
Reshape_173/shapeConst*
dtype0*
_output_shapes
: *
valueB 
d
Reshape_173Reshape
split_6:21Reshape_173/shape*
T0*
Tshape0*
_output_shapes
: 
T
Reshape_174/shapeConst*
_output_shapes
: *
valueB *
dtype0
d
Reshape_174Reshape
split_6:22Reshape_174/shape*
_output_shapes
: *
T0*
Tshape0
T
Reshape_175/shapeConst*
dtype0*
_output_shapes
: *
valueB 
d
Reshape_175Reshape
split_6:23Reshape_175/shape*
Tshape0*
_output_shapes
: *
T0
T
Reshape_176/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_176Reshape
split_6:24Reshape_176/shape*
_output_shapes
: *
T0*
Tshape0
T
Reshape_177/shapeConst*
dtype0*
_output_shapes
: *
valueB 
d
Reshape_177Reshape
split_6:25Reshape_177/shape*
Tshape0*
_output_shapes
: *
T0
b
Reshape_178/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
m
Reshape_178Reshape
split_6:26Reshape_178/shape*
_output_shapes
:	h*
T0*
Tshape0
b
Reshape_179/shapeConst*
dtype0*
_output_shapes
:*
valueB"h      
m
Reshape_179Reshape
split_6:27Reshape_179/shape*
T0*
Tshape0*
_output_shapes
:	h
\
Reshape_180/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_180Reshape
split_6:28Reshape_180/shape*
T0*
Tshape0*
_output_shapes	
:
\
Reshape_181/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_181Reshape
split_6:29Reshape_181/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_182/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_182Reshape
split_6:30Reshape_182/shape* 
_output_shapes
:
*
T0*
Tshape0
b
Reshape_183/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
n
Reshape_183Reshape
split_6:31Reshape_183/shape* 
_output_shapes
:
*
T0*
Tshape0
\
Reshape_184/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_184Reshape
split_6:32Reshape_184/shape*
T0*
Tshape0*
_output_shapes	
:
\
Reshape_185/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_185Reshape
split_6:33Reshape_185/shape*
Tshape0*
_output_shapes	
:*
T0
b
Reshape_186/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_186Reshape
split_6:34Reshape_186/shape*
Tshape0*
_output_shapes
:	*
T0
b
Reshape_187/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_187Reshape
split_6:35Reshape_187/shape*
_output_shapes
:	*
T0*
Tshape0
[
Reshape_188/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_188Reshape
split_6:36Reshape_188/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_189/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_189Reshape
split_6:37Reshape_189/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_190/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_190Reshape
split_6:38Reshape_190/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_191/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_191Reshape
split_6:39Reshape_191/shape*
T0*
Tshape0*
_output_shapes
:
T
Reshape_192/shapeConst*
dtype0*
_output_shapes
: *
valueB 
d
Reshape_192Reshape
split_6:40Reshape_192/shape*
T0*
Tshape0*
_output_shapes
: 
T
Reshape_193/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_193Reshape
split_6:41Reshape_193/shape*
_output_shapes
: *
T0*
Tshape0
T
Reshape_194/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_194Reshape
split_6:42Reshape_194/shape*
T0*
Tshape0*
_output_shapes
: 
T
Reshape_195/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_195Reshape
split_6:43Reshape_195/shape*
_output_shapes
: *
T0*
Tshape0
b
Reshape_196/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
m
Reshape_196Reshape
split_6:44Reshape_196/shape*
_output_shapes
:	h*
T0*
Tshape0
b
Reshape_197/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
m
Reshape_197Reshape
split_6:45Reshape_197/shape*
_output_shapes
:	h*
T0*
Tshape0
\
Reshape_198/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_198Reshape
split_6:46Reshape_198/shape*
_output_shapes	
:*
T0*
Tshape0
\
Reshape_199/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_199Reshape
split_6:47Reshape_199/shape*
Tshape0*
_output_shapes	
:*
T0
b
Reshape_200/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_200Reshape
split_6:48Reshape_200/shape* 
_output_shapes
:
*
T0*
Tshape0
b
Reshape_201/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_201Reshape
split_6:49Reshape_201/shape*
T0*
Tshape0* 
_output_shapes
:

\
Reshape_202/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_202Reshape
split_6:50Reshape_202/shape*
Tshape0*
_output_shapes	
:*
T0
\
Reshape_203/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_203Reshape
split_6:51Reshape_203/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_204/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_204Reshape
split_6:52Reshape_204/shape*
T0*
Tshape0*
_output_shapes
:	
b
Reshape_205/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_205Reshape
split_6:53Reshape_205/shape*
_output_shapes
:	*
T0*
Tshape0
[
Reshape_206/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_206Reshape
split_6:54Reshape_206/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_207/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_207Reshape
split_6:55Reshape_207/shape*
T0*
Tshape0*
_output_shapes
:
b
Reshape_208/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
m
Reshape_208Reshape
split_6:56Reshape_208/shape*
T0*
Tshape0*
_output_shapes
:	h
b
Reshape_209/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
m
Reshape_209Reshape
split_6:57Reshape_209/shape*
T0*
Tshape0*
_output_shapes
:	h
\
Reshape_210/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_210Reshape
split_6:58Reshape_210/shape*
T0*
Tshape0*
_output_shapes	
:
\
Reshape_211/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_211Reshape
split_6:59Reshape_211/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_212/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_212Reshape
split_6:60Reshape_212/shape*
Tshape0* 
_output_shapes
:
*
T0
b
Reshape_213/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_213Reshape
split_6:61Reshape_213/shape*
T0*
Tshape0* 
_output_shapes
:

\
Reshape_214/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_214Reshape
split_6:62Reshape_214/shape*
T0*
Tshape0*
_output_shapes	
:
\
Reshape_215/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_215Reshape
split_6:63Reshape_215/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_216/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_216Reshape
split_6:64Reshape_216/shape*
T0*
Tshape0*
_output_shapes
:	
b
Reshape_217/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_217Reshape
split_6:65Reshape_217/shape*
T0*
Tshape0*
_output_shapes
:	
[
Reshape_218/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_218Reshape
split_6:66Reshape_218/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_219/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_219Reshape
split_6:67Reshape_219/shape*
T0*
Tshape0*
_output_shapes
:
¨
	Assign_21Assignpi/dense/kernelReshape_152*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	h
 
	Assign_22Assignpi/dense/biasReshape_153*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
­
	Assign_23Assignpi/dense_1/kernelReshape_154* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
¤
	Assign_24Assignpi/dense_1/biasReshape_155*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ź
	Assign_25Assignpi/dense_2/kernelReshape_156*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ł
	Assign_26Assignpi/dense_2/biasReshape_157*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(

	Assign_27Assign
pi/log_stdReshape_158*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
¨
	Assign_28Assignvf/dense/kernelReshape_159*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	h*
use_locking(*
T0
 
	Assign_29Assignvf/dense/biasReshape_160*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
­
	Assign_30Assignvf/dense_1/kernelReshape_161*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:

¤
	Assign_31Assignvf/dense_1/biasReshape_162*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ź
	Assign_32Assignvf/dense_2/kernelReshape_163*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
Ł
	Assign_33Assignvf/dense_2/biasReshape_164*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:
¨
	Assign_34Assignvc/dense/kernelReshape_165*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	h
 
	Assign_35Assignvc/dense/biasReshape_166*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:
­
	Assign_36Assignvc/dense_1/kernelReshape_167*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
¤
	Assign_37Assignvc/dense_1/biasReshape_168*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias
Ź
	Assign_38Assignvc/dense_2/kernelReshape_169*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ł
	Assign_39Assignvc/dense_2/biasReshape_170*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ť
	Assign_40Assignpenalty/penalty_paramReshape_171*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
Ą
	Assign_41Assignbeta1_powerReshape_172*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
Ą
	Assign_42Assignbeta2_powerReshape_173*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
°
	Assign_43Assignpenalty/penalty_param/AdamReshape_174*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
˛
	Assign_44Assignpenalty/penalty_param/Adam_1Reshape_175*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
Ł
	Assign_45Assignbeta1_power_1Reshape_176*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(
Ł
	Assign_46Assignbeta2_power_1Reshape_177*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
­
	Assign_47Assignpi/dense/kernel/AdamReshape_178*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	h
Ż
	Assign_48Assignpi/dense/kernel/Adam_1Reshape_179*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	h*
use_locking(*
T0
Ľ
	Assign_49Assignpi/dense/bias/AdamReshape_180*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
§
	Assign_50Assignpi/dense/bias/Adam_1Reshape_181*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
˛
	Assign_51Assignpi/dense_1/kernel/AdamReshape_182*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

´
	Assign_52Assignpi/dense_1/kernel/Adam_1Reshape_183*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Š
	Assign_53Assignpi/dense_1/bias/AdamReshape_184*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ť
	Assign_54Assignpi/dense_1/bias/Adam_1Reshape_185*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(
ą
	Assign_55Assignpi/dense_2/kernel/AdamReshape_186*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
ł
	Assign_56Assignpi/dense_2/kernel/Adam_1Reshape_187*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
¨
	Assign_57Assignpi/dense_2/bias/AdamReshape_188*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
Ş
	Assign_58Assignpi/dense_2/bias/Adam_1Reshape_189*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias

	Assign_59Assignpi/log_std/AdamReshape_190*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
 
	Assign_60Assignpi/log_std/Adam_1Reshape_191*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
˛
	Assign_61Assignpenalty/penalty_param/Adam_2Reshape_192*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
˛
	Assign_62Assignpenalty/penalty_param/Adam_3Reshape_193*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(

	Assign_63Assignbeta1_power_2Reshape_194*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: 

	Assign_64Assignbeta2_power_2Reshape_195*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: 
­
	Assign_65Assignvf/dense/kernel/AdamReshape_196*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	h
Ż
	Assign_66Assignvf/dense/kernel/Adam_1Reshape_197*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	h*
use_locking(
Ľ
	Assign_67Assignvf/dense/bias/AdamReshape_198*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:
§
	Assign_68Assignvf/dense/bias/Adam_1Reshape_199*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vf/dense/bias
˛
	Assign_69Assignvf/dense_1/kernel/AdamReshape_200* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(
´
	Assign_70Assignvf/dense_1/kernel/Adam_1Reshape_201*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Š
	Assign_71Assignvf/dense_1/bias/AdamReshape_202*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias
Ť
	Assign_72Assignvf/dense_1/bias/Adam_1Reshape_203*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias
ą
	Assign_73Assignvf/dense_2/kernel/AdamReshape_204*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
ł
	Assign_74Assignvf/dense_2/kernel/Adam_1Reshape_205*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel
¨
	Assign_75Assignvf/dense_2/bias/AdamReshape_206*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:
Ş
	Assign_76Assignvf/dense_2/bias/Adam_1Reshape_207*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(
­
	Assign_77Assignvc/dense/kernel/AdamReshape_208*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	h*
use_locking(*
T0
Ż
	Assign_78Assignvc/dense/kernel/Adam_1Reshape_209*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	h*
use_locking(
Ľ
	Assign_79Assignvc/dense/bias/AdamReshape_210*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vc/dense/bias
§
	Assign_80Assignvc/dense/bias/Adam_1Reshape_211*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(
˛
	Assign_81Assignvc/dense_1/kernel/AdamReshape_212*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:

´
	Assign_82Assignvc/dense_1/kernel/Adam_1Reshape_213*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Š
	Assign_83Assignvc/dense_1/bias/AdamReshape_214*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias
Ť
	Assign_84Assignvc/dense_1/bias/Adam_1Reshape_215*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(
ą
	Assign_85Assignvc/dense_2/kernel/AdamReshape_216*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
ł
	Assign_86Assignvc/dense_2/kernel/Adam_1Reshape_217*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	
¨
	Assign_87Assignvc/dense_2/bias/AdamReshape_218*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:
Ş
	Assign_88Assignvc/dense_2/bias/Adam_1Reshape_219*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ä
group_deps_6NoOp
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
^Assign_40
^Assign_41
^Assign_42
^Assign_43
^Assign_44
^Assign_45
^Assign_46
^Assign_47
^Assign_48
^Assign_49
^Assign_50
^Assign_51
^Assign_52
^Assign_53
^Assign_54
^Assign_55
^Assign_56
^Assign_57
^Assign_58
^Assign_59
^Assign_60
^Assign_61
^Assign_62
^Assign_63
^Assign_64
^Assign_65
^Assign_66
^Assign_67
^Assign_68
^Assign_69
^Assign_70
^Assign_71
^Assign_72
^Assign_73
^Assign_74
^Assign_75
^Assign_76
^Assign_77
^Assign_78
^Assign_79
^Assign_80
^Assign_81
^Assign_82
^Assign_83
^Assign_84
^Assign_85
^Assign_86
^Assign_87
^Assign_88
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_16ecf0f6cb764edeb8ca2ce86e0feaa7/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
ő
save/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
î
save/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ę
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
ř
save/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
ń
save/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
â
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ś
save/AssignAssignbeta1_powersave/RestoreV2*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(
Ź
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param
¤
save/Assign_2Assignbeta1_power_2save/RestoreV2:2*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ş
save/Assign_3Assignbeta2_powersave/RestoreV2:3*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(
Ź
save/Assign_4Assignbeta2_power_1save/RestoreV2:4*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
¤
save/Assign_5Assignbeta2_power_2save/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: 
´
save/Assign_6Assignpenalty/penalty_paramsave/RestoreV2:6*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param
š
save/Assign_7Assignpenalty/penalty_param/Adamsave/RestoreV2:7*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
ť
save/Assign_8Assignpenalty/penalty_param/Adam_1save/RestoreV2:8*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
ť
save/Assign_9Assignpenalty/penalty_param/Adam_2save/RestoreV2:9*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
˝
save/Assign_10Assignpenalty/penalty_param/Adam_3save/RestoreV2:10*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
Ť
save/Assign_11Assignpi/dense/biassave/RestoreV2:11*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
°
save/Assign_12Assignpi/dense/bias/Adamsave/RestoreV2:12* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
˛
save/Assign_13Assignpi/dense/bias/Adam_1save/RestoreV2:13*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ł
save/Assign_14Assignpi/dense/kernelsave/RestoreV2:14*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	h
¸
save/Assign_15Assignpi/dense/kernel/Adamsave/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	h
ş
save/Assign_16Assignpi/dense/kernel/Adam_1save/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	h
Ż
save/Assign_17Assignpi/dense_1/biassave/RestoreV2:17*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(
´
save/Assign_18Assignpi/dense_1/bias/Adamsave/RestoreV2:18*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(
ś
save/Assign_19Assignpi/dense_1/bias/Adam_1save/RestoreV2:19*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
¸
save/Assign_20Assignpi/dense_1/kernelsave/RestoreV2:20*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
˝
save/Assign_21Assignpi/dense_1/kernel/Adamsave/RestoreV2:21*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ż
save/Assign_22Assignpi/dense_1/kernel/Adam_1save/RestoreV2:22*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ž
save/Assign_23Assignpi/dense_2/biassave/RestoreV2:23*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ł
save/Assign_24Assignpi/dense_2/bias/Adamsave/RestoreV2:24*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ľ
save/Assign_25Assignpi/dense_2/bias/Adam_1save/RestoreV2:25*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ˇ
save/Assign_26Assignpi/dense_2/kernelsave/RestoreV2:26*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
ź
save/Assign_27Assignpi/dense_2/kernel/Adamsave/RestoreV2:27*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
ž
save/Assign_28Assignpi/dense_2/kernel/Adam_1save/RestoreV2:28*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
¤
save/Assign_29Assign
pi/log_stdsave/RestoreV2:29*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Š
save/Assign_30Assignpi/log_std/Adamsave/RestoreV2:30*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ť
save/Assign_31Assignpi/log_std/Adam_1save/RestoreV2:31*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ť
save/Assign_32Assignvc/dense/biassave/RestoreV2:32*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(
°
save/Assign_33Assignvc/dense/bias/Adamsave/RestoreV2:33* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
˛
save/Assign_34Assignvc/dense/bias/Adam_1save/RestoreV2:34*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(
ł
save/Assign_35Assignvc/dense/kernelsave/RestoreV2:35*
validate_shape(*
_output_shapes
:	h*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel
¸
save/Assign_36Assignvc/dense/kernel/Adamsave/RestoreV2:36*
validate_shape(*
_output_shapes
:	h*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel
ş
save/Assign_37Assignvc/dense/kernel/Adam_1save/RestoreV2:37*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	h
Ż
save/Assign_38Assignvc/dense_1/biassave/RestoreV2:38*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
´
save/Assign_39Assignvc/dense_1/bias/Adamsave/RestoreV2:39*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:
ś
save/Assign_40Assignvc/dense_1/bias/Adam_1save/RestoreV2:40*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias
¸
save/Assign_41Assignvc/dense_1/kernelsave/RestoreV2:41*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
˝
save/Assign_42Assignvc/dense_1/kernel/Adamsave/RestoreV2:42*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:

ż
save/Assign_43Assignvc/dense_1/kernel/Adam_1save/RestoreV2:43*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ž
save/Assign_44Assignvc/dense_2/biassave/RestoreV2:44*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ł
save/Assign_45Assignvc/dense_2/bias/Adamsave/RestoreV2:45*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ľ
save/Assign_46Assignvc/dense_2/bias/Adam_1save/RestoreV2:46*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save/Assign_47Assignvc/dense_2/kernelsave/RestoreV2:47*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
ź
save/Assign_48Assignvc/dense_2/kernel/Adamsave/RestoreV2:48*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
ž
save/Assign_49Assignvc/dense_2/kernel/Adam_1save/RestoreV2:49*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel
Ť
save/Assign_50Assignvf/dense/biassave/RestoreV2:50*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:
°
save/Assign_51Assignvf/dense/bias/Adamsave/RestoreV2:51* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
˛
save/Assign_52Assignvf/dense/bias/Adam_1save/RestoreV2:52*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:
ł
save/Assign_53Assignvf/dense/kernelsave/RestoreV2:53*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	h
¸
save/Assign_54Assignvf/dense/kernel/Adamsave/RestoreV2:54*
_output_shapes
:	h*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(
ş
save/Assign_55Assignvf/dense/kernel/Adam_1save/RestoreV2:55*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	h
Ż
save/Assign_56Assignvf/dense_1/biassave/RestoreV2:56*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
´
save/Assign_57Assignvf/dense_1/bias/Adamsave/RestoreV2:57*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(
ś
save/Assign_58Assignvf/dense_1/bias/Adam_1save/RestoreV2:58*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save/Assign_59Assignvf/dense_1/kernelsave/RestoreV2:59*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˝
save/Assign_60Assignvf/dense_1/kernel/Adamsave/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ż
save/Assign_61Assignvf/dense_1/kernel/Adam_1save/RestoreV2:61*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ž
save/Assign_62Assignvf/dense_2/biassave/RestoreV2:62*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:
ł
save/Assign_63Assignvf/dense_2/bias/Adamsave/RestoreV2:63*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(
ľ
save/Assign_64Assignvf/dense_2/bias/Adam_1save/RestoreV2:64*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save/Assign_65Assignvf/dense_2/kernelsave/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
ź
save/Assign_66Assignvf/dense_2/kernel/Adamsave/RestoreV2:66*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel
ž
save/Assign_67Assignvf/dense_2/kernel/Adam_1save/RestoreV2:67*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	
	
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard "&<
save/Const:0save/Identity:0save/restore_all (5 @F8"đ
trainable_variablesŘŐ
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
R
pi/log_std:0pi/log_std/Assignpi/log_std/read:02pi/log_std/initial_value:08
s
vf/dense/kernel:0vf/dense/kernel/Assignvf/dense/kernel/read:02,vf/dense/kernel/Initializer/random_uniform:08
b
vf/dense/bias:0vf/dense/bias/Assignvf/dense/bias/read:02!vf/dense/bias/Initializer/zeros:08
{
vf/dense_1/kernel:0vf/dense_1/kernel/Assignvf/dense_1/kernel/read:02.vf/dense_1/kernel/Initializer/random_uniform:08
j
vf/dense_1/bias:0vf/dense_1/bias/Assignvf/dense_1/bias/read:02#vf/dense_1/bias/Initializer/zeros:08
{
vf/dense_2/kernel:0vf/dense_2/kernel/Assignvf/dense_2/kernel/read:02.vf/dense_2/kernel/Initializer/random_uniform:08
j
vf/dense_2/bias:0vf/dense_2/bias/Assignvf/dense_2/bias/read:02#vf/dense_2/bias/Initializer/zeros:08
s
vc/dense/kernel:0vc/dense/kernel/Assignvc/dense/kernel/read:02,vc/dense/kernel/Initializer/random_uniform:08
b
vc/dense/bias:0vc/dense/bias/Assignvc/dense/bias/read:02!vc/dense/bias/Initializer/zeros:08
{
vc/dense_1/kernel:0vc/dense_1/kernel/Assignvc/dense_1/kernel/read:02.vc/dense_1/kernel/Initializer/random_uniform:08
j
vc/dense_1/bias:0vc/dense_1/bias/Assignvc/dense_1/bias/read:02#vc/dense_1/bias/Initializer/zeros:08
{
vc/dense_2/kernel:0vc/dense_2/kernel/Assignvc/dense_2/kernel/read:02.vc/dense_2/kernel/Initializer/random_uniform:08
j
vc/dense_2/bias:0vc/dense_2/bias/Assignvc/dense_2/bias/read:02#vc/dense_2/bias/Initializer/zeros:08
~
penalty/penalty_param:0penalty/penalty_param/Assignpenalty/penalty_param/read:02%penalty/penalty_param/initial_value:08"$
train_op

Adam
Adam_1
Adam_2"çA
	variablesŮAÖA
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
R
pi/log_std:0pi/log_std/Assignpi/log_std/read:02pi/log_std/initial_value:08
s
vf/dense/kernel:0vf/dense/kernel/Assignvf/dense/kernel/read:02,vf/dense/kernel/Initializer/random_uniform:08
b
vf/dense/bias:0vf/dense/bias/Assignvf/dense/bias/read:02!vf/dense/bias/Initializer/zeros:08
{
vf/dense_1/kernel:0vf/dense_1/kernel/Assignvf/dense_1/kernel/read:02.vf/dense_1/kernel/Initializer/random_uniform:08
j
vf/dense_1/bias:0vf/dense_1/bias/Assignvf/dense_1/bias/read:02#vf/dense_1/bias/Initializer/zeros:08
{
vf/dense_2/kernel:0vf/dense_2/kernel/Assignvf/dense_2/kernel/read:02.vf/dense_2/kernel/Initializer/random_uniform:08
j
vf/dense_2/bias:0vf/dense_2/bias/Assignvf/dense_2/bias/read:02#vf/dense_2/bias/Initializer/zeros:08
s
vc/dense/kernel:0vc/dense/kernel/Assignvc/dense/kernel/read:02,vc/dense/kernel/Initializer/random_uniform:08
b
vc/dense/bias:0vc/dense/bias/Assignvc/dense/bias/read:02!vc/dense/bias/Initializer/zeros:08
{
vc/dense_1/kernel:0vc/dense_1/kernel/Assignvc/dense_1/kernel/read:02.vc/dense_1/kernel/Initializer/random_uniform:08
j
vc/dense_1/bias:0vc/dense_1/bias/Assignvc/dense_1/bias/read:02#vc/dense_1/bias/Initializer/zeros:08
{
vc/dense_2/kernel:0vc/dense_2/kernel/Assignvc/dense_2/kernel/read:02.vc/dense_2/kernel/Initializer/random_uniform:08
j
vc/dense_2/bias:0vc/dense_2/bias/Assignvc/dense_2/bias/read:02#vc/dense_2/bias/Initializer/zeros:08
~
penalty/penalty_param:0penalty/penalty_param/Assignpenalty/penalty_param/read:02%penalty/penalty_param/initial_value:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0

penalty/penalty_param/Adam:0!penalty/penalty_param/Adam/Assign!penalty/penalty_param/Adam/read:02.penalty/penalty_param/Adam/Initializer/zeros:0

penalty/penalty_param/Adam_1:0#penalty/penalty_param/Adam_1/Assign#penalty/penalty_param/Adam_1/read:020penalty/penalty_param/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
|
pi/dense/kernel/Adam:0pi/dense/kernel/Adam/Assignpi/dense/kernel/Adam/read:02(pi/dense/kernel/Adam/Initializer/zeros:0

pi/dense/kernel/Adam_1:0pi/dense/kernel/Adam_1/Assignpi/dense/kernel/Adam_1/read:02*pi/dense/kernel/Adam_1/Initializer/zeros:0
t
pi/dense/bias/Adam:0pi/dense/bias/Adam/Assignpi/dense/bias/Adam/read:02&pi/dense/bias/Adam/Initializer/zeros:0
|
pi/dense/bias/Adam_1:0pi/dense/bias/Adam_1/Assignpi/dense/bias/Adam_1/read:02(pi/dense/bias/Adam_1/Initializer/zeros:0

pi/dense_1/kernel/Adam:0pi/dense_1/kernel/Adam/Assignpi/dense_1/kernel/Adam/read:02*pi/dense_1/kernel/Adam/Initializer/zeros:0

pi/dense_1/kernel/Adam_1:0pi/dense_1/kernel/Adam_1/Assignpi/dense_1/kernel/Adam_1/read:02,pi/dense_1/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_1/bias/Adam:0pi/dense_1/bias/Adam/Assignpi/dense_1/bias/Adam/read:02(pi/dense_1/bias/Adam/Initializer/zeros:0

pi/dense_1/bias/Adam_1:0pi/dense_1/bias/Adam_1/Assignpi/dense_1/bias/Adam_1/read:02*pi/dense_1/bias/Adam_1/Initializer/zeros:0

pi/dense_2/kernel/Adam:0pi/dense_2/kernel/Adam/Assignpi/dense_2/kernel/Adam/read:02*pi/dense_2/kernel/Adam/Initializer/zeros:0

pi/dense_2/kernel/Adam_1:0pi/dense_2/kernel/Adam_1/Assignpi/dense_2/kernel/Adam_1/read:02,pi/dense_2/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_2/bias/Adam:0pi/dense_2/bias/Adam/Assignpi/dense_2/bias/Adam/read:02(pi/dense_2/bias/Adam/Initializer/zeros:0

pi/dense_2/bias/Adam_1:0pi/dense_2/bias/Adam_1/Assignpi/dense_2/bias/Adam_1/read:02*pi/dense_2/bias/Adam_1/Initializer/zeros:0
h
pi/log_std/Adam:0pi/log_std/Adam/Assignpi/log_std/Adam/read:02#pi/log_std/Adam/Initializer/zeros:0
p
pi/log_std/Adam_1:0pi/log_std/Adam_1/Assignpi/log_std/Adam_1/read:02%pi/log_std/Adam_1/Initializer/zeros:0

penalty/penalty_param/Adam_2:0#penalty/penalty_param/Adam_2/Assign#penalty/penalty_param/Adam_2/read:020penalty/penalty_param/Adam_2/Initializer/zeros:0

penalty/penalty_param/Adam_3:0#penalty/penalty_param/Adam_3/Assign#penalty/penalty_param/Adam_3/read:020penalty/penalty_param/Adam_3/Initializer/zeros:0
\
beta1_power_2:0beta1_power_2/Assignbeta1_power_2/read:02beta1_power_2/initial_value:0
\
beta2_power_2:0beta2_power_2/Assignbeta2_power_2/read:02beta2_power_2/initial_value:0
|
vf/dense/kernel/Adam:0vf/dense/kernel/Adam/Assignvf/dense/kernel/Adam/read:02(vf/dense/kernel/Adam/Initializer/zeros:0

vf/dense/kernel/Adam_1:0vf/dense/kernel/Adam_1/Assignvf/dense/kernel/Adam_1/read:02*vf/dense/kernel/Adam_1/Initializer/zeros:0
t
vf/dense/bias/Adam:0vf/dense/bias/Adam/Assignvf/dense/bias/Adam/read:02&vf/dense/bias/Adam/Initializer/zeros:0
|
vf/dense/bias/Adam_1:0vf/dense/bias/Adam_1/Assignvf/dense/bias/Adam_1/read:02(vf/dense/bias/Adam_1/Initializer/zeros:0

vf/dense_1/kernel/Adam:0vf/dense_1/kernel/Adam/Assignvf/dense_1/kernel/Adam/read:02*vf/dense_1/kernel/Adam/Initializer/zeros:0

vf/dense_1/kernel/Adam_1:0vf/dense_1/kernel/Adam_1/Assignvf/dense_1/kernel/Adam_1/read:02,vf/dense_1/kernel/Adam_1/Initializer/zeros:0
|
vf/dense_1/bias/Adam:0vf/dense_1/bias/Adam/Assignvf/dense_1/bias/Adam/read:02(vf/dense_1/bias/Adam/Initializer/zeros:0

vf/dense_1/bias/Adam_1:0vf/dense_1/bias/Adam_1/Assignvf/dense_1/bias/Adam_1/read:02*vf/dense_1/bias/Adam_1/Initializer/zeros:0

vf/dense_2/kernel/Adam:0vf/dense_2/kernel/Adam/Assignvf/dense_2/kernel/Adam/read:02*vf/dense_2/kernel/Adam/Initializer/zeros:0

vf/dense_2/kernel/Adam_1:0vf/dense_2/kernel/Adam_1/Assignvf/dense_2/kernel/Adam_1/read:02,vf/dense_2/kernel/Adam_1/Initializer/zeros:0
|
vf/dense_2/bias/Adam:0vf/dense_2/bias/Adam/Assignvf/dense_2/bias/Adam/read:02(vf/dense_2/bias/Adam/Initializer/zeros:0

vf/dense_2/bias/Adam_1:0vf/dense_2/bias/Adam_1/Assignvf/dense_2/bias/Adam_1/read:02*vf/dense_2/bias/Adam_1/Initializer/zeros:0
|
vc/dense/kernel/Adam:0vc/dense/kernel/Adam/Assignvc/dense/kernel/Adam/read:02(vc/dense/kernel/Adam/Initializer/zeros:0

vc/dense/kernel/Adam_1:0vc/dense/kernel/Adam_1/Assignvc/dense/kernel/Adam_1/read:02*vc/dense/kernel/Adam_1/Initializer/zeros:0
t
vc/dense/bias/Adam:0vc/dense/bias/Adam/Assignvc/dense/bias/Adam/read:02&vc/dense/bias/Adam/Initializer/zeros:0
|
vc/dense/bias/Adam_1:0vc/dense/bias/Adam_1/Assignvc/dense/bias/Adam_1/read:02(vc/dense/bias/Adam_1/Initializer/zeros:0

vc/dense_1/kernel/Adam:0vc/dense_1/kernel/Adam/Assignvc/dense_1/kernel/Adam/read:02*vc/dense_1/kernel/Adam/Initializer/zeros:0

vc/dense_1/kernel/Adam_1:0vc/dense_1/kernel/Adam_1/Assignvc/dense_1/kernel/Adam_1/read:02,vc/dense_1/kernel/Adam_1/Initializer/zeros:0
|
vc/dense_1/bias/Adam:0vc/dense_1/bias/Adam/Assignvc/dense_1/bias/Adam/read:02(vc/dense_1/bias/Adam/Initializer/zeros:0

vc/dense_1/bias/Adam_1:0vc/dense_1/bias/Adam_1/Assignvc/dense_1/bias/Adam_1/read:02*vc/dense_1/bias/Adam_1/Initializer/zeros:0

vc/dense_2/kernel/Adam:0vc/dense_2/kernel/Adam/Assignvc/dense_2/kernel/Adam/read:02*vc/dense_2/kernel/Adam/Initializer/zeros:0

vc/dense_2/kernel/Adam_1:0vc/dense_2/kernel/Adam_1/Assignvc/dense_2/kernel/Adam_1/read:02,vc/dense_2/kernel/Adam_1/Initializer/zeros:0
|
vc/dense_2/bias/Adam:0vc/dense_2/bias/Adam/Assignvc/dense_2/bias/Adam/read:02(vc/dense_2/bias/Adam/Initializer/zeros:0

vc/dense_2/bias/Adam_1:0vc/dense_2/bias/Adam_1/Assignvc/dense_2/bias/Adam_1/read:02*vc/dense_2/bias/Adam_1/Initializer/zeros:0*Ď
serving_defaultť
)
x$
Placeholder:0˙˙˙˙˙˙˙˙˙h$
v
vf/Squeeze:0˙˙˙˙˙˙˙˙˙%
vc
vc/Squeeze:0˙˙˙˙˙˙˙˙˙%
pi
pi/add:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict