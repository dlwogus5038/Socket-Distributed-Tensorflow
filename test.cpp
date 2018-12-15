#define BOOST_PYTHON_STATIC_LIB
#define BOOST_LIB_NAME "boost_numpy3"
#include <boost/config/auto_link.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/container/flat_set.hpp>
#include <iostream>
 
namespace py = boost::python;
namespace np = boost::python::numpy;
 
int main()
{    
    using namespace std;
 
    try
    {
 
        Py_Initialize();
        np::initialize();
 
        PyRun_SimpleString("#-*- coding: utf-8 -*-");        
        py::object main_module = py::import("__main__");
        py::object main_namespace = main_module.attr("__dict__");
        py::object sys_ = py::import("sys");
        
        PyRun_SimpleString("import sys\n"
            "sys.argv = ['']");
 
        py::object print = py::import("__main__").attr("__builtins__").attr("print");
 
        const py::object tf_ = py::import("tensorflow");
        py::exec("import tensorflow as tf", main_namespace);
        
        const py::object input_data_ = py::import("tensorflow.examples.tutorials.mnist").attr("input_data");        
        py::object mnist = input_data_.attr("read_data_sets")("MNIST_data", true);
                
        py::exec("from tensorflow.examples.tutorials.mnist import input_data", main_namespace);
        py::exec("mnist = input_data.read_data_sets('MNIST_data /', one_hot=True)", main_namespace);
                
        //Set variables
        py::object x = py::eval("tf.placeholder(tf.float32, [None, 784])", main_namespace);
        py::object W = py::eval("tf.Variable(tf.zeros([784, 10]))", main_namespace);
        py::object b = py::eval("tf.Variable(tf.zeros([10]))", main_namespace);
        py::object y = tf_.attr("nn").attr("softmax")(tf_.attr("matmul")(x, W) + b);        
 
        //Set Cross-Entropy Model
        py::object y_ = py::eval("tf.placeholder(tf.float32, [None, 10])", main_namespace);
        
        py::object sum = tf_.attr("reduce_sum")(y_ * tf_.attr("log")(y), 1);
        py::object cross_entropy = tf_.attr("reduce_mean")(-1 * sum);
        py::object train_step = tf_.attr("train").attr("GradientDescentOptimizer")(0.5).attr("minimize")(cross_entropy);
 
        // Learn model using SGD
        py::object init = tf_.attr("initialize_all_variables")();
        py::object sess = tf_.attr("Session")();
        sess.attr("run")(init);
 
        for (int i = 0; i < 1000; i++)
        {    
            py::object batches = py::eval("mnist.train.next_batch(100)", main_namespace);
            py::object batch_xs = batches[0];
            py::object batch_ys = batches[1];            
 
            py::dict feed_dict;
            feed_dict[x] = batch_xs;
            feed_dict[y_] = batch_ys;
 
            sess.attr("run")(train_step, feed_dict);        
        }
            
        // Print Accuracy
        py::object f32 = tf_.attr("float32");
        py::object correct_prediction = tf_.attr("equal")(tf_.attr("argmax")(y, 1), tf_.attr("argmax")(y_, 1));
        py::object cast = tf_.attr("cast")(correct_prediction, f32);
        py::object accuracy = tf_.attr("reduce_mean")(cast);
 
        py::dict feed_dict;
        py::object test_x = py::eval("mnist.test.images", main_namespace); 
        py::object test_y = py::eval("mnist.test.labels", main_namespace);
 
        feed_dict[x] = test_x;
        feed_dict[y_] = test_y;
 
        print(sess.attr("run")(accuracy, feed_dict));
    }
    catch (py::error_already_set&)
    {
        PyErr_Print();
        system("pause");
    }
 
    return 0;
}
