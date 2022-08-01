# liveProject: Deep Learning Recommender System

Bridge the gap between the recommender system theory you’ve learned and the hands-on experience you need. This liveProject series provides a deep dive into real-world data management in industrial applications that’s truly rare in learning resources. Developing real-world recommendation systems is much more than understanding how to connect neurons in a neural network and knowing different types of architectures. The true complexity of these systems lies in understanding how to design them to fit real-time use on industry servers, taking into account ranking of items, splitting to offline and online parts, and, most importantly, performing exploration. The advanced, comprehensive liveProjects in this series will provide data scientists with insights—usually learned on the job!—that will take their careers to the next level.

## Project 1 Data Preprocessing

In this liveProject, you’ll design a movie recommendation system step by step, from initial development all the way to a production-ready system. You’ll begin with preparing the data, then you’ll analyze the data, and finally, you’ll preprocess and export it. Along the way, you’ll gain a firm understanding of the data and your users, which is key for developing a system that makes appropriate recommendations. While this liveProject doesn’t contain any machine learning, going through these steps will prepare you for feature engineering and hyperparameter tuning later on.

## Project 2 Two Towers with TensorFlow Recommenders

Once you’ve handled the data, the real magic can begin! In this liveProject, you’ll implement a basic recommendation system using the TensorFlow Recommenders framework—designed specifically for this purpose. First, you’ll calculate four baselines that your future models will have to beat. Next, you’ll learn the basics of developing a model using TensorFlow Recommenders, then design a simple, two-feature Two Towers model. Lastly, you’ll enhance this simple model by adding all the features you created in the previous liveProject while maintaining model performance that beats your four established baselines.

## Project 3 Real-World Inference

Behind the scenes on websites like Amazon, Netflix, and Spotify, models make predictions on thousands of items every day. Then, based on what they’ve learned, they choose only the best recommendations to display for every individual user. In the real world, performing thousands of predictions one by one, as in a notebook-only model, would be highly inefficient. In this liveProject, you’ll reconfigure the models you implemented in the previous project to accept a list of items for each user and then evaluate all items at once—choosing the best recommendations much more quickly and efficiently.


## Project 4 Retrieval and Ranking

Real-world recommendation systems work with millions of users and billions of items. To handle this massive scale, recommendation systems tend to be divided into two types of models (retrieval and ranking), running one after the other, narrowing down the set of items each time. In this liveProject, you’ll implement a retrieval model using TensorFlow Recommenders, combine it with a ranking model, then create a fully functional pipeline going through both models, using a Feature Store. Finally, you’ll explore the scenario where you can’t (or choose not to) run both retrieval and ranking models online in real-time, leveraging the Feature Store once more to use the retrieval model offline.

## Project 5 Feedback-Loop and Exploration Methods

Since recommender systems train and learn over the data they recommended themselves, they will never train over, learn, or recommend items that they didn’t already recommend for some reason, such as insufficient ranking to be seen by the user. It’s important to break this Feedback Loop in order to ensure that suitable recommendations aren’t missed. But you must strike a balance between deviating (just enough) from the system’s predictions through exploration and not defeating the system’s purpose altogether. In this liveProject, you’ll learn three methods of exploration that help you provide better recommendations to your users, as well as the costs and benefits of each.

## Install

Create a virtual environment with Python 3.9 and install from git:

```bash
pip install git+https://github.com/chris-santiago/liveProject-deepRecommender.git
```

---------------
From [Manning liveProjects](https://www.manning.com/liveprojectseries/deep-learning-recommender-system-ser)
