# **Geatpy2 高性能的进化算法框架 ** 
The Genetic and Evolutionary Algorithm Toolbox for Python with high performance.

[教程](http://geatpy.com/index.php/geatpy%E6%95%99%E7%A8%8B/)

[神经进化](https://github.com/Ewenwan/ENNEoS/blob/master/README.md)

进化算法(Evolutionary Algorithm, EA)是一类通过模拟自然界生物自然选择和自然进化的随机搜索算法。与传统搜索算法如二分法、斐波那契法、牛顿法、抛物线法等相比，进化算法有着高鲁棒性和求解高度复杂的非线性问题(如NP完全问题)的能力。在过去的40年中，进化算法得到了不同的发展，现主要有三类：1)主要由美国J. H. Holland提出的的遗传算法(Genetic Algorithm, GA)；2)主要由德国I. Rechenberg提出的进化策略(Evolution strategies, ES)；3)主要由美国的L. J. Fogel提出的进化规划(Evolutionary Programming, EP)。

Geatpy是一个高性能实用型进化算法工具箱，提供了许多已实现的进化算法各项操作的函数，如初始化种群、选择、交叉、变异、多目标优化参考点生成、非支配排序、多目标优化GD、IGD、HV等指标的计算等等。没有繁杂的深度封装，可以清晰地看到其基本结构及相关算法的实现，并利用Geatpy函数方便地进行自由组合，实现和研究多种改进的进化算法、多目标优化、并行进化算法等，解决传统优化算法难以解决的问题。Geatpy在Python上提供简便易用的面向对象的进化算法框架。通过继承问题类，可以轻松把实际问题与进化算法相结合。Geatpy还提供多种进化算法模板类，涵盖遗传算法、差分进化算法、进化策略、多目标进化优化算法等，可以通过调用算法模板，轻松解决单目标优化、多目标优化、组合优化、约束优化等问题。这些都是开源的，你可以参照这些模板，加以改进或重构，以便实现效果更好的进化算法，或是让进化算法更好地与当前正在进行的项目进行融合。

程序结构：

其中“core”文件夹里面全部为Geatpy工具箱的内核函数；

“templates”文件夹存放的是Geatpy的进化算法模板；

“testbed”是进化算法的测试平台，内含多种单目标优化、多目标优化、组合优化的测试集。

“demo”文件夹中包含了应用Geatpy工具箱求解问题的案例。

“operators”是2.2.2版本之后新增的，里面存放着面向对象的重组和变异算子类，这些重组和变异算子类通过

调用“core”文件夹下的重组和变异算子来执行相关的操作。

Geatpy的面向对象进化算法框架有四个大类：Algorithm(算法模板顶级父类)、Pop-ulation(种群类)、PsyPopulation(多染色体种群类)和Problem(问题类)，分别存放在“Al-gorithm.py”、“Population.py”、“Problem.py”文件中。

Geatpy2整体上看由工具箱内核函数（内核层）和面向对象进化算法框架（框架层）两部分组成。其中面向对象进化算法框架主要有四个大类：Problem问题类、Algorithm算法模板类、Population种群类和PsyPopulation多染色体种群类。

Problem 类定义了与问题相关的一些信息，如问题名称name、优化目标的维数M、决策变量的个数Dim、决策变量的范围ranges、决策变量的边界borders 等。maxormins是一个记录着各个目标函数是最小化抑或是最大化的行向量，其中元素为1 表示对应的目标是最小化目标；为-1 表示对应的是最大化目标。例如M=3，maxormins=np.array([1,-1,1])，此时表示有三个优化目标，其中第一、第三个是最小化目标，第二个是最大化目标。varTypes 是一个记录着决策变量类型的行向量，其中的元素为0 表示对应的决策变量是连续型变量；为1 表示对应的是离散型变量。待求解的目标函数定义在aimFunc() 的函数中。calBest() 函数则用于从理论上计算目标函数的理论最优值。在实际使用时，不是直接在Problem 类的文件中修改相关代码使用的，而是通过定义一个继承Problem的子类来完成对问题的定义的。这些在后面的章节中会详细讲述。对于Problem 类中各属性的详细含义可查看Problem.py 源码。

Population 类是一个表示种群的类。一个种群包含很多个个体，而每个个体都有一条染色体(若要用多染色体，则使用多个种群、并把每个种群对应个体关联起来即可)。除了染色体外，每个个体都有一个译码矩阵Field(或俗称区域描述器) 来标识染色体应该如何解码得到表现型，同时也有其对应的目标函数值以及适应度。种群类就是一个把所有个体的这些数据统一存储起来的一个类。比如里面的Chrom 是一个存储种群所有个体染色体的矩阵，它的每一行对应一个个体的染色体；ObjV 是一个目标函数值矩阵，每一行对应一个个体的所有目标函数值，每一列对应一个目标。对于Population 类中各属性的详细含义可查看Population.py 源码以及下一章“Geatpy 数据结构”。

PsyPopulation类是继承了Population的支持多染色体混合编码的种群类。一个种群包含很多个个体，而每个个体都有多条染色体。用Chroms列表存储所有的染色体矩阵(Chrom)；Encodings列表存储各染色体对应的编码方式(Encoding)；Fields列表存储各染色体对应的译码矩阵(Field)。

Algorithm 类是进化算法的核心类。它既存储着跟进化算法相关的一些参数，同时也在其继承类中实现具体的进化算法。比如Geatpy 中的moea_NSGA3_templet.py 是实现了多目标优化NSGA-III 算法的进化算法模板类，它是继承了Algorithm 类的具体算法的模板类。关于Algorithm 类中各属性的含义可以查看Algorithm.py 源码。这些算法模板通过调用Geatpy 工具箱提供的进化算法库函数实现对种群的进化操作，同时记录进化过程中的相关信息

**特色

1. Geatpy是一个功能强大的进化算法工具箱，并提供耦合度很低的进化算法框架。没有过于抽象的复杂封装，入门门槛低。它提供多种格式的编码方式以及丰富的选择、交叉和变异算子。你可以在极短的时间里掌握Geatpy的用法，并把Geatpy融合到你正在进行的项目或实际问题的解决方案中。

2. Geatpy的一大特色是提供众多自由、清晰的进化算法模板。在模板中，你可以清晰地看到算法的完整流程，更加掌握遗传算法的更多细节。通过修改进化算法模板，你可以轻松解决难以想象多的优化问题。可以将Geatpy用作研究进化算法的通用测试平台，实现各种改进的进化算法。

3.利用Numpy+mkl以及C内核实现高性能计算，使得Geatpy在算法执行效率上有着不俗的表现。在不失通用性的保证下在速度上比采用Java或是Matlab编写的或是采用C++内核及Python接口编写的进化算法工具箱、框架和平台快得多。

4.提供丰富的进化过程追踪分析功能，可以看到在进化过程中决策空间和目标空间的变化，以及进化过程中各项评价指标的变化，以帮助用户更好地研究进化算法。

5.支持轻松实现约束优化，可通过罚函数法或者利用可行性法则来完成对复杂约束条件(包括不等式约束和等式约束)的处理。

6.提供非常详尽的代码注释，是目前注释量最多的进化算法工具箱。让用户可以最快速度地上手进化算法、研究进化算法和应用进化算法。

7.支持多染色体混合编码的进化优化，使得Geatpy可以轻松应对各种复杂问题。



[单目标优化案例：](https://github.com/geatpy-dev/geatpy/tree/master/geatpy/demo/soea_demo)

[多目标优化案例：](https://github.com/geatpy-dev/geatpy/tree/master/geatpy/demo/moea_demo)

[标准测试平台（包含单目标、多目标、超多目标、旅行商问题等的测试集](https://github.com/geatpy-dev/geatpy/tree/master/geatpy/testbed)

![Travis](https://travis-ci.org/geatpy-dev/geatpy.svg?branch=master)
[![Package Status](https://img.shields.io/pypi/status/geatpy.svg)](https://pypi.org/project/geatpy/)
[![License](https://img.shields.io/pypi/l/geatpy.svg)](https://github.com/geatpy-dev/geatpy/blob/master/LICENSE)
![Python](https://img.shields.io/badge/python->=3.5-green.svg)
![Pypi](https://img.shields.io/badge/pypi-2.2.3-blue.svg)

## Introduction
* **Website (including documentation)**: http://www.geatpy.com
* **Demo** : https://github.com/geatpy-dev/geatpy/tree/master/geatpy/demo
* **Pypi page** : https://pypi.org/project/geatpy/
* **Contact us**: http://geatpy.com/index.php/about/
* **Bug reports**: https://github.com/geatpy-dev/geatpy/issues
* **Notice**: http://geatpy.com/index.php/notice/
* **FAQ**: http://geatpy.com/index.php/faq/

Geatpy provides:

* global optimization capabilities in **Python** using genetic and other evolutionary algorithms to solve problems unsuitable for traditional optimization approaches.

* a great many of **evolutionary operators**, so that you can deal with **single, multiple and many objective optimization** problems.

## Improvement of Geatpy 2.2.3

* Improve the performance of crtpp and ranking.

* Rebuild the core of NSGA-II, NSGA-III and RVEA to get higher performances.

* Add new multi-objective optimization test problem: TNK.

## Installation
1.Installing online:

    pip install geatpy

2.From source:

    python setup.py install

or

    pip install <filename>.whl

**Attention**: Geatpy requires numpy>=1.16.0, matplotlib>=3.0.0 and scipy>=1.0.0, the installation program won't help you install them so that you have to install both of them by yourselves.

## Versions

**Geatpy** must run under **Python**3.5, 3.6 or 3.7 in Windows x32/x64, Linux x64 or Mac OS x64.

There are different versions for **Windows**, **Linux** and **Mac**, you can download them from http://geatpy.com/

The version of **Geatpy** on github is the latest version suitable for **Python** >= 3.5

You can also **update** Geatpy by executing the command:

    pip install --upgrade geatpy

If something wrong happened, such as decoding error about 'utf8' of pip, run this command instead or execute it as an administrator:

    pip install --upgrade --user geatpy

Quick start
-----------

Here is the UML figure of Geatpy2.

![image](https://github.com/geatpy-dev/geatpy/blob/master/structure.png)

For solving a multi-objective optimization problem, you can use **Geatpy** mainly in two steps:

1.Write down the aim function and some relevant settings in a derivative class named **MyProblem**, which is inherited from **Problem** class:

```python
"""MyProblem.py"""
import numpy as np
import geatpy as ea
class MyProblem(ea.Problem): # Inherited from Problem class.
    def __init__(self, M): # M is the number of objects.
        name = 'DTLZ1' # Problem's name.
        maxormins = [1] * M # All objects are need to be minimized.
        Dim = M + 4 # Set the dimension of decision variables.
        varTypes = [0] * Dim # Set the types of decision variables. 0 means continuous while 1 means discrete.
        lb = [0] * Dim # The lower bound of each decision variable.
        ub = [1] * Dim # The upper bound of each decision variable.
        lbin = [1] * Dim # Whether the lower boundary is included.
        ubin = [1] * Dim # Whether the upper boundary is included.
        # Call the superclass's constructor to complete the instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    def aimFunc(self, pop): # Write the aim function here, pop is an object of Population class.
        Vars = pop.Phen # Get the decision variables
        XM = Vars[:,(self.M-1):]
        g = np.array([100 * (self.Dim - self.M + 1 + np.sum(((XM - 0.5)**2 - np.cos(20 * np.pi * (XM - 0.5))), 1))]).T
        ones_metrix = np.ones((Vars.shape[0], 1))
        pop.ObjV = 0.5 * np.fliplr(np.cumprod(np.hstack([ones_metrix, Vars[:,:self.M-1]]), 1)) * np.hstack([ones_metrix, 1 - Vars[:, range(self.M - 2, -1, -1)]]) * np.tile(1 + g, (1, self.M))
    def calBest(self): # Calculate the theoretic global optimal solution here.
        uniformPoint, ans = ea.crtup(self.M, 10000) # create 10000 uniform points.
        realBestObjV = uniformPoint / 2
        return realBestObjV
```

2.Instantiate **MyProblem** class and a derivative class inherited from **Algorithm** class in a Python script file "main.py" then execute it. **For example**, trying to find the pareto front of **DTLZ1**, do as the following:

```python
"""main.py"""
import geatpy as ea # Import geatpy
from MyProblem import MyProblem # Import MyProblem class
"""=========================Instantiate your problem=========================="""
M = 3                      # Set the number of objects.
problem = MyProblem(M)     # Instantiate MyProblem class
"""===============================Population set=============================="""
Encoding = 'RI'            # Encoding type.
NIND = 100                 # Set the number of individuals.
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # Create the field descriptor.
population = ea.Population(Encoding, Field, NIND) # Instantiate Population class(Just instantiate, not initialize the population yet.)
"""================================Algorithm set==============================="""
myAlgorithm = ea.moea_NSGA3_templet(problem, population) # Instantiate a algorithm class.
myAlgorithm.MAXGEN = 500 # Set the max times of iteration.
"""===============================Start evolution=============================="""
NDSet = myAlgorithm.run() # Run the algorithm templet.
"""=============================Analyze the result============================="""
PF = problem.calBest() # Get the global pareto front.
GD = ea.indicator.GD(NDSet.ObjV, PF) # Calculate GD
IGD = ea.indicator.IGD(NDSet.ObjV, PF) # Calculate IGD
HV = ea.indicator.HV(NDSet.ObjV, PF) # Calculate HV
Space = ea.indicator.spacing(NDSet.ObjV) # Calculate Space
print('The number of non-dominated result: %s'%(NDSet.sizes))
print('GD: ',GD)
print('IGD: ',IGD)
print('HV: ', HV)
print('Space: ', Space)
```

Run the "main.py" and the result is:

![image](https://github.com/geatpy-dev/geatpy/blob/master/geatpy/testbed/moea_test/moea_test_DTLZ/Pareto%20Front.svg)

The number of non-dominated result: 91

GD:  0.00019492736742063313

IGD:  0.02058320808720775

HV:  0.8413590788841248

Space:  0.00045742613969278813

For solving another problem: **Ackley-30D**, which has only one object and 30 decision variables, what you need to do is almost the same as above.

1.Write the aim function in "MyProblem.py".

```python
import numpy as np
import geatpy as ea
class Ackley(ea.Problem): # Inherited from Problem class.
    def __init__(self, D = 30):
        name = 'Ackley' # Problem's name.
        M = 1 # Set the number of objects.
        maxormins = [1] * M # All objects are need to be minimized.
        Dim = D # Set the dimension of decision variables.
        varTypes = [0] * Dim # Set the types of decision variables. 0 means continuous while 1 means discrete.
        lb = [-32.768] * Dim # The lower bound of each decision variable.
        ub = [32.768] * Dim # The upper bound of each decision variable.
        lbin = [1] * Dim # Whether the lower boundary is included.
        ubin = [1] * Dim # Whether the upper boundary is included.
        # Call the superclass's constructor to complete the instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    def aimFunc(self, pop): # Write the aim function here, pop is an object of Population class.
        x = pop.Phen # Get the decision variables
        n = self.Dim
        f = np.array([-20 * np.exp(-0.2*np.sqrt(1/n*np.sum(x**2, 1))) - np.exp(1/n * np.sum(np.cos(2 * np.pi * x), 1)) + np.e + 20]).T
        return f, CV
    def calBest(self): # Calculate the global optimal solution here.
        realBestObjV = np.array([[0]])
        return realBestObjV
```

2.Write "main.py" to execute the algorithm templet to solve the problem.

```python
import geatpy as ea # import geatpy
import numpy as np
from MyProblem import Ackley
"""=========================Instantiate your problem=========================="""
problem = Ackley(30) # Instantiate MyProblem class.
"""===============================Population set=============================="""
Encoding = 'RI'                # Encoding type.
NIND = 20                      # Set the number of individuals.
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # Create the field descriptor.
population = ea.Population(Encoding, Field, NIND) # Instantiate Population class(Just instantiate, not initialize the population yet.)
"""================================Algorithm set==============================="""
myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population) # Instantiate a algorithm class.
myAlgorithm.MAXGEN = 1000      # Set the max times of iteration.
myAlgorithm.mutOper.F = 0.5    # Set the F of DE
myAlgorithm.recOper.XOVR = 0.2 # Set the Cr of DE (Here it is marked as XOVR)
myAlgorithm.drawing = 1 # 1 means draw the figure of the result
"""===============================Start evolution=============================="""
[population, obj_trace, var_trace] = myAlgorithm.run() # Run the algorithm templet.
"""=============================Analyze the result============================="""
best_gen = np.argmin(obj_trace[:, 1]) # Get the best generation.
best_ObjV = np.min(obj_trace[:, 1])
print('The objective value of the best solution is: %s'%(best_ObjV))
print('Effective iteration times: %s'%(obj_trace.shape[0]))
print('The best generation is: %s'%(best_gen + 1))
print('The number of evolution is: %s'%(myAlgorithm.evalsNum))
```

The result is:

![image](https://github.com/geatpy-dev/geatpy/blob/master/geatpy/testbed/soea_test/soea_test_Ackley/result1.svg)

The objective value of the best solution is: 5.8686921988737595e-09

Effective iteration times: 1000

The best generation is: 1000

The number of evolution is: 20000

To get more tutorials, please link to http://www.geatpy.com.
