import tensorflow as tf
import numpy as np

import fileIO as IO
import cal_loss as loss
import vgg16
import matplotlib.pyplot as plt


# ## Style-Transfer Algorithm

def style_transfer(content_image, style_image,
                   content_layer_ids, style_layer_ids,
                   weight_content=1.5, weight_style=10.0,
                   weight_denoise=0.3,
                   num_iterations=120, step_size=10.0):
    """
    Use gradient descent to find an image that minimizes the
    loss-functions of the content-layers and style-layers. 
    """

    # Create an instance of the VGG16-model. 
    model = vgg16.VGG16()

    # Create a TensorFlow-session.
    session = tf.InteractiveSession(graph=model.graph)

    # create a log file
    summary_writer=tf.summary.FileWriter('log')
    summary_writer.add_graph(session.graph)
    

    # Print the names of the content-layers.
    print("Content layers:")
    print(model.get_layer_names(content_layer_ids))
    print()

    # Print the names of the style-layers.
    print("Style layers:")
    print(model.get_layer_names(style_layer_ids))
    print()

    with tf.name_scope('content_loss'):
        loss_content = (loss.create_content_loss(session=session,
                                       model=model,
                                       content_image=content_image,
                                       layer_ids=content_layer_ids))
        tf.summary.scalar('content_loss',loss_content)

    with tf.name_scope('style_loss'):
        loss_style = loss.create_style_loss(session=session,
                                   model=model,
                                   style_image=style_image,
                                   layer_ids=style_layer_ids)
        tf.summary.scalar('style_loss',loss_style)

    with tf.name_scope('style_loss'):         
        loss_denoise = loss.create_denoise_loss(model)
        tf.summary.scalar('denoise_loss',loss_denoise)


    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')


    # Initialize the adjustment values for the loss-functions.
    session.run([adj_content.initializer,
                 adj_style.initializer,
                 adj_denoise.initializer])

    # add 1e-10 to avoid the possibility of division by zero.
    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))


    loss_combined = (weight_content * adj_content * loss_content + 
    		    weight_style * adj_style * loss_style +                  
    		    weight_denoise * adj_denoise * loss_denoise)

    # gradient decent 
    gradient = tf.gradients(loss_combined, model.input)

    # run in each optimization iteration.
    run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]

    mixed_image = np.random.rand(*content_image.shape) + 128

    content_loss=[]
    style_loss=[]
    denoise_loss=[]
    grad_list=[]
    
    for i in range(num_iterations):
        # Create a feed-dict with the mixed-image.
        feed_dict = model.create_feed_dict(image=mixed_image)

        grad, adj_content_val, adj_style_val, adj_denoise_val=(session.run(run_list,
        feed_dict=feed_dict))
        content_loss.append(adj_content_val)
        style_loss.append(adj_style_val)
        denoise_loss.append(adj_denoise_val)
        grad_list.append(grad)
        

        # Reduce the dimensionality of the gradient.
        grad = np.squeeze(grad)

        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        mixed_image -= grad * step_size_scaled
        mixed_image = np.clip(mixed_image, 0.0, 255.0)

        print(". ", end="")

        # Display status every 10 iterations
        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)

            # Print adjustment weights for loss-functions.
            msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
            print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))


            # Plot the content-, style- and mixed-images.
            IO.plot_images(content_image=content_image,
                        style_image=style_image,
                        mixed_image=mixed_image)
            
    print()
    print("Final image:")
    IO.plot_image_big(mixed_image)

    steps=range(num_iterations)
    plt.plot(steps,content_loss,'b',label='content_loss')
    plt.plot(steps,style_loss,'r',label='style_loss')
    plt.show()

    plt.plot(steps,denoise_loss,'k',label='denoise_loss')
    plt.show()
    

    session.close()
    return mixed_image
