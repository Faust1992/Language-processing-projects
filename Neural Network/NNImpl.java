/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public Node outputNode=null;// single output node that represents the result of the regression
	
	public ArrayList<Instance> trainingSet=null;//the training set
	
	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs
	
	
	/**
 	* This constructor creates the nodes necessary for the neural network
 	* Also connects the nodes of different layers
 	* After calling the constructor the last node of both inputNodes and  
 	* hiddenNodes will be bias nodes. 
 	*/
	
	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;
		
		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		int outputNodeCount=1;
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}
		
		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);
		
		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}
		
		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);
			


		Node node=new Node(4);
		//Connecting output node with hidden layer nodes
		for(int j=0;j<hiddenNodes.size();j++)
		{
			NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[j]);
			node.parents.add(nwp);
		}	
		outputNode = node;
			
	}
	
	/**
	 * Get the output from the neural network for a single instance. That is, set the values of the training instance to
	the appropriate input nodes, percolate them through the network, then return the activation value at the single output
	node. This is your estimate of y. 
	 */
	
	public double calculateOutputForInstance(Instance inst)
	{
		if (inst == null)
		return -1;
		for (int i = 0; i < inputNodes.size() - 1; i++)
			inputNodes.get(i).setInput(inst.attributes.get(i));
		for (int i = 0; i < hiddenNodes.size() - 1; i++)
			hiddenNodes.get(i).calculateOutput();
		outputNode.calculateOutput();
		return outputNode.getOutput();
	}
	

	
	
	
	/**
	 * Trains a neural network with the parameters initialized in the constructor
	 * The parameters are stored as attributes of this class
	 */
	
	public void train()
	{
		for (int i = 0; i < maxEpoch; i++){
			for (int j = 0; j < trainingSet.size(); j++){
				double out = calculateOutputForInstance(trainingSet.get(j));
				double g;
				if (out > 0)
					g = 1;
				else 
					g = 0;
				double err = trainingSet.get(j).output - out;
				double dk = err*g;
				for (int k = 0; k < outputNode.parents.size(); k++){
					double out2 = outputNode.parents.get(k).node.getOutput();
					double g2;
					if (out2 > 0)
						g2 = 1;
					else 
						g2 = 0;
					double dj = g2*dk*outputNode.parents.get(k).weight;
                    if (k != outputNode.parents.size()-1)
					for (int m = 0; m < outputNode.parents.get(k).node.parents.size(); m++)
						outputNode.parents.get(k).node.parents.get(m).weight+= learningRate*outputNode.parents.get(k).node.parents.get(m).node.getOutput()*dj;
					outputNode.parents.get(k).weight+= learningRate*out2*dk;
				}
			}
		}
	}
	/**
	 * Returns the mean squared error of a dataset. That is, the sum of the squared error (T-O) for each instance
	in the dataset divided by the number of instances in the dataset.
	 */
	

	public double getMeanSquaredError(List<Instance> dataset){
		if (dataset == null)
		return -1;
		double sum = 0;
		for (int i = 0; i < dataset.size(); i++)
			sum+= Math.pow(dataset.get(i).output - calculateOutputForInstance(dataset.get(i)),2);
		sum = sum/dataset.size();
		return sum;
	}
}
