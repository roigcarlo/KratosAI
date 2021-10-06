import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

def expression(x):
    y = x * x
    return y

def scenario_1():
    ''' Runs a expression inside a tape '''
    print("Running scenario 1...")
    x = tf.constant(3.0)

    with tf.GradientTape() as gt:
        gt.watch(x)
        y = expression(x)
        dy_dx = gt.gradient(y, x)
        print("Gradient 1 of expression:", dy_dx)

def scenario_2():
    ''' Runs a expression inside a tape inside another tape '''
    print("Running scenario 2...")
    x = tf.constant(3.0)

    with tf.GradientTape() as gt1:
        gt1.watch(x)
        with tf.GradientTape() as gt2:
            gt2.watch(x)
            y = expression(x)
            dy_dx = gt2.gradient(y, x)
        d2y_d2x = gt1.gradient(dy_dx, x)
        print("Gradient 1 of expression:", dy_dx)
        print("Gradient 2 of expression:", d2y_d2x)

def scenario_3():
    ''' Runs a expression in between two tapes '''
    print("Running scenario 3...")
    x = tf.constant(3.0)

    with tf.GradientTape() as gt1:
        gt1.watch(x)
        y = expression(x)
        with tf.GradientTape() as gt2:
            gt2.watch(x)
            dy_dx = gt2.gradient(y, x)
        print("Gradient 1 of expression:", dy_dx)
        try:
            d2y_d2x = gt1.gradient(dy_dx, x)
            print("Gradient 2 of expression:", d2y_d2x)
        except Exception as e:
            print("Error calculating d2y_d2x:", e)

def scenario_4():
    ''' Runs a expression before a tape '''
    print("Running scenario 4...")
    x = tf.constant(3.0)

    y = expression(x)
    with tf.GradientTape() as gt2:
        gt2.watch(x)
        dy_dx = gt2.gradient(y, x)
    print("Gradient 1 of expression:", dy_dx)

def scenario_5():
    ''' Runs a expression after a tape '''
    print("Running scenario 5...")
    x = tf.constant(3.0)

    with tf.GradientTape() as gt2:
        gt2.watch(x)
    
    y = expression(x)
    try:
        dy_dx = gt2.gradient(y, x)
    except Exception as e:
        print("Error calculating dy_dx:", e)
    print("Gradient 1 of expression:", dy_dx)

scenario_1() # exp in tape                      - Ok
scenario_2() # exp in tape in tape              - Ok
scenario_3() # exp in tape and tape after exp   - Fail
scenario_4() # exp before tape                  - Fail
scenario_5() # exp after tape                   - Fail

# opt = tf.keras.optimizers.Adam(learning_rate=0.1)
# var1 = tf.Variable(10.0)

# def f(x) = x**2

# def loss(x):
#     output = f(x)
#     with tf.tape as tape:
#         grad = tape.gradient(out, x) 

#     return output + grad * grad
# loss = lambda: (f(var1) tape.grad(f(var(1),var1)))/2.0       # d(loss)/d(var1) == var1
# step_count = opt.minimize(loss, [var1]).numpy()
# # The first step is `-learning_rate*sign(grad)`
# var1.numpy()
