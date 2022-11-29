## Import Libaries -------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt 
from time import time
import tensorflow as tf
import scipy.io as sp

from MDOF_DeepONet_Dataset import Training_Dataset
from MDOF_DeepONet_Model import UnstackedMultiOutputDeepONet  
from MDOF_DeepONet_Loss import Loss     

## Define Parameters ------------------------------------------------------------

Training_Data = False    # To generate new training data, set Training_data = True
Testing_Data = False     # To generate new testing data, set Testing_data = True
plotResult = True        # To plot the displacement of a random input signal, plotResult = True
plotLosses = True        # To plot the losses, set plotLosses = True

Num_Train = 1500         # Number of training samples
ft_Train = 20            # Fourier Transformation
Num_Test = 10000         # Number of test samples
ft_Test = 20
pointspsamples = 25      # Number of points per training sample

dt = 0.01                # time step
min_t = 0                # starting time
max_t = 2                # ending time

Optimizer_Type = "Adam" 
Learning_Rate = 1e-03
Epochs = 200
Batch_Size = 128
Initializer = 'Glorot Normal'

Layers_Branch = [40, 40]
Layers_Trunk = [40, 40]
Activations_Trunk = ['relu', 'relu']
Activations_Branch = ['relu', 'relu']

# ----------------------------------------------------------------------------------
if Training_Data:

    am_start = -10     # Range of the amplitudes
    am_end = 10      
    fr_start = 0       # Range of the frequencies
    fr_end = 10

    M1 = 10            # Mass 1
    M2 = 10            # Mass 2
    M3 = 9             # Mass 3
    M4 = 9             # Mass 4
    M5 = 7.5           # Mass 5
    alpha = 100;
    C1 = 100           # Damping Coefficient 1
    C2 = 100           # Damping Coefficient 2
    C3 = 90            # Damping Coefficient 3
    C4 = 90            # Damping Coefficient 4
    C5 = 75            # Damping Coefficient 5
    K1 = 10000         # Stiffness Coefficient 1
    K2 = 10000         # Stiffness Coefficient 2
    K3 = 9000          # Stiffness Coefficient 3
    K4 = 9000          # Stiffness Coefficient 4
    K5 = 7500          # Stiffness Coefficient 5

    x_init = 0.01      # initial displacement
    x_dot_init = 0.05  # initial velocity

    Training_Dataset(Num_Train, min_t, max_t, dt, ft_Train, am_start, am_end, fr_start, fr_end, M1, M2, M3, M4, M5, alpha, C1, C2, C3, C4, C5, K1, K2, K3, K4, K5, x_init, x_dot_init)

data_train = sp.loadmat('data_5DOF_duffing_FT'+str(ft_Train)+'_Samples'+str(Num_Train)+'.mat')

if Testing_Data:

    am_start = -10     # Range of the amplitudes
    am_end = 10      
    fr_start = 0       # Range of the frequencies
    fr_end = 10

    M1 = 10            # Mass 1
    M2 = 10            # Mass 2
    M3 = 9             # Mass 3
    M4 = 9             # Mass 4
    M5 = 7.5           # Mass 5
    alpha = 100;
    C1 = 100           # Damping Coefficient 1
    C2 = 100           # Damping Coefficient 2
    C3 = 90            # Damping Coefficient 3
    C4 = 90            # Damping Coefficient 4
    C5 = 75            # Damping Coefficient 5
    K1 = 10000         # Stiffness Coefficient 1
    K2 = 10000         # Stiffness Coefficient 2
    K3 = 9000          # Stiffness Coefficient 3
    K4 = 9000          # Stiffness Coefficient 4
    K5 = 7500          # Stiffness Coefficient 5

    x_init = 0.01      # initial displacement
    x_dot_init = 0.05  # initial velocity

    Training_Dataset(Num_Test, min_t, max_t, dt, ft_Test, am_start, am_end, fr_start, fr_end, M1, M2, M3, M4, M5, alpha, C1, C2, C3, C4, C5, K1, K2, K3, K4, K5, x_init, x_dot_init)

data_test = sp.loadmat('data_5DOF_duffing_FT'+str(ft_Test)+'_Samples'+str(Num_Test)+'.mat')

## Model, Optimizer and loss ---------------------------------------------------

model = UnstackedMultiOutputDeepONet(layer_sizes_branch=Layers_Branch,
                layer_sizes_trunk=Layers_Trunk,
                activation_branch=Activations_Branch,
                activation_trunk=Activations_Trunk,
                kernel_initializer=Initializer)
    
if Optimizer_Type == "Adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_Rate)
elif Optimizer_Type == "SGD":
    optimizer = tf.keras.optimizers.SGD(learning_rate=Learning_Rate)
elif Optimizer_Type == "RMSProp":
    optimizer = tf.keras.optimizers.RMSProp(learning_rate=Learning_Rate)
else:
     optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_Rate)
     print("Adam optimizer was chosen as default setting!")

loss_fn = Loss()


## Creating Input Data ----------------------------------------------------------------

f_train = data_train['f']
f_test = data_test['f']

t = np.arange(min_t, max_t, dt)
lt = t.shape[0]

y_true_random_1 = np.zeros([Num_Train*pointspsamples,1])
y_true_random_2 = np.zeros([Num_Train*pointspsamples,1])
y_true_random_3 = np.zeros([Num_Train*pointspsamples,1])
y_true_random_4 = np.zeros([Num_Train*pointspsamples,1])
y_true_random_5 = np.zeros([Num_Train*pointspsamples,1])

pointer = np.random.randint(0,lt-1,Num_Train*pointspsamples)

y_true_test_1 = data_test['y1']
y_true_test_2 = data_test['y2']
y_true_test_3 = data_test['y3']
y_true_test_4 = data_test['y4']
y_true_test_5 = data_test['y5']

for disp_num in range (0,5):
    if disp_num == 0:
        y = data_train['y1']
        for i in range(0,pointspsamples):
            for j in range(0,Num_Train):
                y_true_random_1[int(Num_Train*i+j),0] = y[j,pointer[Num_Train*i+j]]
    elif disp_num == 1:
        y = data_train['y2']
        for i in range(0,pointspsamples):
            for j in range(0,Num_Train):
                y_true_random_2[int(Num_Train*i+j),0] = y[j,pointer[Num_Train*i+j]]
    elif disp_num == 2:
        y = data_train['y3']
        for i in range(0,pointspsamples):
            for j in range(0,Num_Train):
                y_true_random_3[int(Num_Train*i+j),0] = y[j,pointer[Num_Train*i+j]]
    elif disp_num == 3:
        y = data_train['y4']
        for i in range(0,pointspsamples):
            for j in range(0,Num_Train):
                y_true_random_4[int(Num_Train*i+j),0] = y[j,pointer[Num_Train*i+j]]
    else:
        y = data_train['y5']
        for i in range(0,pointspsamples):
            for j in range(0,Num_Train):
                y_true_random_5[int(Num_Train*i+j),0] = y[j,pointer[Num_Train*i+j]]
    
   
force =  np.tile(f_train[0:Num_Train,0:-1:2],(pointspsamples,1))
t_random = np.zeros([Num_Train*pointspsamples,1])


force_test = np.zeros([Num_Test*lt,100])
t_test = np.zeros([Num_Test*lt,1])
y_pred_test_1 = np.zeros([Num_Test,lt])
y_pred_test_2 = np.zeros([Num_Test,lt])
y_pred_test_3 = np.zeros([Num_Test,lt])
y_pred_test_4 = np.zeros([Num_Test,lt])
y_pred_test_5 = np.zeros([Num_Test,lt])

for i in range(0,pointspsamples):
        for j in range(0,Num_Train):
            t_random[int(Num_Train*i+j),0] = t[pointer[Num_Train*i+j]] 
            
x_test = f_test[:,0:-1:2]

for i in range(0,Num_Test):
    force_test[lt*i:lt+lt*i,:] = np.tile(x_test[i,:],[lt,1])
    t_test[lt*i:lt+lt*i,:] = t.reshape([lt,1]) 

y_true_random_1 = y_true_random_1.astype('float32')
y_true_random_2 = y_true_random_2.astype('float32')
y_true_random_3 = y_true_random_3.astype('float32')
y_true_random_4 = y_true_random_4.astype('float32')
y_true_random_5 = y_true_random_5.astype('float32')

force = force.astype('float32')
t_random = t_random.astype('float32')

force_test = force_test.astype('float32')
t_test = t_test.astype('float32')
    
train_dataset = tf.data.Dataset.from_tensor_slices((force, t_random, y_true_random_1, y_true_random_2, y_true_random_3, y_true_random_4, y_true_random_5))
train_dataset = train_dataset.shuffle(buffer_size=Num_Train).batch(Batch_Size, drop_remainder=True)
# train_dataset = train_dataset.batch(Batch_Size, drop_remainder=True)

# Training ------------------------------------------------------------------------

@tf.function
def train_step(force_batch,t_random_batch, y_true_random_1_batch,  y_true_random_2_batch,  y_true_random_3_batch,  y_true_random_4_batch,  y_true_random_5_batch):
    with tf.GradientTape(persistent=True) as tape:
        [y_pred_batch_1, y_pred_batch_2, y_pred_batch_3, y_pred_batch_4, y_pred_batch_5] = model.call(force_batch, t_random_batch, training=True)
        loss_value = loss_fn.call(y_true_random_1_batch,  y_true_random_2_batch,  y_true_random_3_batch,  y_true_random_4_batch,  y_true_random_5_batch, y_pred_batch_1, y_pred_batch_2, y_pred_batch_3, y_pred_batch_4, y_pred_batch_5)
    grads = tape.gradient(loss_value, model.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

@tf.function
def test_step(force_test, t_test):
    with tf.GradientTape(persistent=True) as tape:
        [y_pred_test_new_1, y_pred_test_new_2, y_pred_test_new_3, y_pred_test_new_4, y_pred_test_new_5] = model.call(force_test, t_test, training=False)
    return y_pred_test_new_1, y_pred_test_new_2, y_pred_test_new_3, y_pred_test_new_4, y_pred_test_new_5

print('---Start training ---')
startTraining = time()
history = []

for epoch in range(Epochs):
    for step, (force_batch,t_random_batch, y_true_random_1_batch,  y_true_random_2_batch,  y_true_random_3_batch,  y_true_random_4_batch,  y_true_random_5_batch) in enumerate(train_dataset):
        loss_value = train_step(force_batch, t_random_batch, y_true_random_1_batch,  y_true_random_2_batch,  y_true_random_3_batch,  y_true_random_4_batch,  y_true_random_5_batch)
    
    [y_pred_test_new_1, y_pred_test_new_2, y_pred_test_new_3, y_pred_test_new_4, y_pred_test_new_5] = test_step(force_test, t_test)
    y_pred_test_new_1 = np.array(y_pred_test_new_1)
    y_pred_test_new_2 = np.array(y_pred_test_new_2)
    y_pred_test_new_3 = np.array(y_pred_test_new_3)
    y_pred_test_new_4 = np.array(y_pred_test_new_4)
    y_pred_test_new_5 = np.array(y_pred_test_new_5)
    history.append([loss_value.numpy()])

endTraining = time() - startTraining
print(f'Time taken for training: {endTraining} s, {endTraining/60} min, {endTraining/60/60} h')

# Results --------------------------------------------------------------------------------------------------

if plotLosses:
    epoch_count = []
    for i in range(1,len(history)+1):
        epoch_count.append(i)
    history = np.array(history)
    fig_loss, ax = plt.subplots()
    plt.plot(epoch_count, history[:,0], 'r--')
    plt.legend(['Training Loss'])
    ax.set_yscale("log", nonpositive='clip')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Loss" )
    plt.savefig('Loss.png') 
            
    
if plotResult:

    for i in range(0,Num_Test):
        y_pred_test_1[i,:] = y_pred_test_new_1[lt*i:lt+lt*i].reshape(1,-1)
        y_pred_test_2[i,:] = y_pred_test_new_2[lt*i:lt+lt*i].reshape(1,-1)
        y_pred_test_3[i,:] = y_pred_test_new_3[lt*i:lt+lt*i].reshape(1,-1)
        y_pred_test_4[i,:] = y_pred_test_new_4[lt*i:lt+lt*i].reshape(1,-1)
        y_pred_test_5[i,:] = y_pred_test_new_5[lt*i:lt+lt*i].reshape(1,-1)
    
    # Displacement 1
    plt.figure(figsize = [20,20])
    plt.plot(t,y_true_test_1[7,:], 'b--', label = 'Ground Truth')
    plt.plot(t,y_pred_test_1[7,:],'r-', label = 'DeepONet')
                    
    plt.xlabel('Time [s]',fontsize = 20)
    plt.ylabel("$x_{"+ str(disp_num+1) + "}$ [m]",fontsize = 20)
    plt.title("Predicted Displacement for Displacement "+str(disp_num+1))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('Predicted_Displacement_1.png') 
    
    plt.figure(figsize = [20,20])
    mean_true = np.mean(y_true_test_1, axis=0)
    mean_pred = np.mean(y_pred_test_1, axis=0)

    plt.plot(t,mean_true, 'b--', label = 'Mean (Ground Truth)')
    plt.plot(t,mean_pred,'r-', label = 'Mean (DeepONet)')

    var_true = np.var(y_true_test_1, axis=0)
    var_pred = np.var(y_pred_test_1, axis=0)

    plt.plot(t,var_true, 'b-.', label = 'Var (Ground Truth)')
    plt.plot(t,var_pred,'r:', label = 'Var (DeepONet)')
            
    plt.xlabel('Time [s]',fontsize = 20)
    plt.ylabel('$\mu_{x1} (m) , \sigma^2_{x1} (m^2)$',fontsize = 20)
    plt.title("Predicted Mean/Variance for FT "+str(ft_Test)+", Displacement 1 ")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('Mean_Variance_Displacement_1.png')
    
    # Displacement 2
    plt.figure(figsize = [20,20])
    plt.plot(t,y_true_test_2[7,:], 'b--', label = 'Ground Truth')
    plt.plot(t,y_pred_test_2[7,:],'r-', label = 'DeepONet')
                    
    plt.xlabel('Time [s]',fontsize = 20)
    plt.ylabel("$x_1$ [m]",fontsize = 20)
    plt.title("Predicted Displacement for Displacement 2")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('Predicted_Displacement_2.png') 

    plt.figure(figsize = [20,20])
    mean_true = np.mean(y_true_test_2, axis=0)
    mean_pred = np.mean(y_pred_test_2, axis=0)

    plt.plot(t,mean_true, 'b--', label = 'Mean (Ground Truth)')
    plt.plot(t,mean_pred,'r-', label = 'Mean (DeepONet)')

    var_true = np.var(y_true_test_2, axis=0)
    var_pred = np.var(y_pred_test_2, axis=0)

    plt.plot(t,var_true, 'b-.', label = 'Var (Ground Truth)')
    plt.plot(t,var_pred,'r:', label = 'Var (DeepONet)')
            
    plt.xlabel('Time [s]',fontsize = 20)
    plt.ylabel('$\mu_{x2} (m) , \sigma^2_{x2} (m^2)$',fontsize = 20)
    plt.title("Predicted Mean/Variance for FT "+str(ft_Test)+", Displacement 2 ")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('Mean_Variance_Displacement_2.png')
    
    # Displacement 3
    plt.figure(figsize = [20,20])
    plt.plot(t,y_true_test_3[7,:], 'b--', label = 'Ground Truth')
    plt.plot(t,y_pred_test_3[7,:],'r-', label = 'DeepONet')
                    
    plt.xlabel('Time [s]',fontsize = 20)
    plt.ylabel("$x_3$ [m]",fontsize = 20)
    plt.title("Predicted Displacement for Displacement 3")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('Predicted_Displacement_3.png') 

    plt.figure(figsize = [20,20])
    mean_true = np.mean(y_true_test_3, axis=0)
    mean_pred = np.mean(y_pred_test_3, axis=0)

    plt.plot(t,mean_true, 'b--', label = 'Mean (Ground Truth)')
    plt.plot(t,mean_pred,'r-', label = 'Mean (DeepONet)')

    var_true = np.var(y_true_test_3, axis=0)
    var_pred = np.var(y_pred_test_3, axis=0)

    plt.plot(t,var_true, 'b-.', label = 'Var (Ground Truth)')
    plt.plot(t,var_pred,'r:', label = 'Var (DeepONet)')
            
    plt.xlabel('Time [s]',fontsize = 20)
    plt.ylabel('$\mu_{x3} (m) , \sigma^2_{x3} (m^2)$',fontsize = 20)
    plt.title("Predicted Mean/Variance for FT "+str(ft_Test)+", Displacement 3 ")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('Mean_Variance_Displacement_3.png')
    
    # Displacement 4
    plt.figure(figsize = [20,20])
    plt.plot(t,y_true_test_4[7,:], 'b--', label = 'Ground Truth')
    plt.plot(t,y_pred_test_4[7,:],'r-', label = 'DeepONet')
                    
    plt.xlabel('Time [s]',fontsize = 20)
    plt.ylabel("$x_4$ [m]",fontsize = 20)
    plt.title("Predicted Displacement for Displacement 4")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('Predicted_Displacement_4.png') 

    plt.figure(figsize = [20,20])
    mean_true = np.mean(y_true_test_4, axis=0)
    mean_pred = np.mean(y_pred_test_4, axis=0)

    plt.plot(t,mean_true, 'b--', label = 'Mean (Ground Truth)')
    plt.plot(t,mean_pred,'r-', label = 'Mean (DeepONet)')

    var_true = np.var(y_true_test_4, axis=0)
    var_pred = np.var(y_pred_test_4, axis=0)

    plt.plot(t,var_true, 'b-.', label = 'Var (Ground Truth)')
    plt.plot(t,var_pred,'r:', label = 'Var (DeepONet)')
            
    plt.xlabel('Time [s]',fontsize = 20)
    plt.ylabel('$\mu_{x4} (m) , \sigma^2_{x4} (m^2)$',fontsize = 20)
    plt.title("Predicted Mean/Variance for FT "+str(ft_Test)+", Displacement 4 ")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('Mean_Variance_Displacement_4.png')
    
    # Displacement 5
    plt.figure(figsize = [20,20])
    plt.plot(t,y_true_test_5[7,:], 'b--', label = 'Ground Truth')
    plt.plot(t,y_pred_test_5[7,:],'r-', label = 'DeepONet')
                    
    plt.xlabel('Time [s]',fontsize = 20)
    plt.ylabel("$x_5$ [m]",fontsize = 20)
    plt.title("Predicted Displacement for Displacement 5")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('Predicted_Displacement_5.png') 
    
    plt.figure(figsize = [20,20])
    mean_true = np.mean(y_true_test_5, axis=0)
    mean_pred = np.mean(y_pred_test_5, axis=0)

    plt.plot(t,mean_true, 'b--', label = 'Mean (Ground Truth)')
    plt.plot(t,mean_pred,'r-', label = 'Mean (DeepONet)')

    var_true = np.var(y_true_test_5, axis=0)
    var_pred = np.var(y_pred_test_5, axis=0)

    plt.plot(t,var_true, 'b-.', label = 'Var (Ground Truth)')
    plt.plot(t,var_pred,'r:', label = 'Var (DeepONet)')
            
    plt.xlabel('Time [s]',fontsize = 20)
    plt.ylabel('$\mu_{x5} (m) , \sigma^2_{x5} (m^2)$',fontsize = 20)
    plt.title("Predicted Mean/Variance for FT "+str(ft_Test)+", Displacement 5 ")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('Mean_Variance_Displacement_5.png')