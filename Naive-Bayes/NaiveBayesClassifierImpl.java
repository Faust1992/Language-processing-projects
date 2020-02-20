import java.util.HashMap;
import java.util.Map;
import java.util.HashSet;
/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {

	//THESE VARIABLES ARE OPTIONAL TO USE, but HashMaps will make your life much, much easier on this assignment.

	//dictionaries of form word:frequency that store the number of times word w has been seen in documents of type label
	//for example, comedyCounts["mirth"] should store the total number of "mirth" tokens that appear in comedy documents
   private HashMap<String, Integer> tragedyCounts = new HashMap<String, Integer>();
   private HashMap<String, Integer> comedyCounts = new HashMap<String, Integer>();
   private HashMap<String, Integer> historyCounts = new HashMap<String, Integer>();
   
   //prior probabilities, ie. P(T), P(C), and P(H)
   //use the training set for the numerator and denominator
   private double tragedyPrior;
   private double comedyPrior;
   private double historyPrior;
   
   //total number of word TOKENS for each type of document in the training set, ie. the sum of the length of all documents with a given label
   private int tTokenSum;
   private int cTokenSum;
   private int hTokenSum;
   private int tDocumentSum;
   private int cDocumentSum;
   private int hDocumentSum;
   //full vocabulary, update in training, cardinality is necessary for smoothing
   private HashSet<String> vocabulary = new HashSet<String>();
   Instance[] training;

  /**
   * Trains the classifier with the provided training data
   Should iterate through the training instances, and, for each word in the documents, update the variables above appropriately.
   The dictionary of frequencies and prior probabilites can then be used at classification time.
   */
  @Override
  public void train(Instance[] trainingData) {
	  training = trainingData;
	  tTokenSum = 0;
	  cTokenSum = 0;
	  hTokenSum = 0;
	  tDocumentSum = 0;
	  cDocumentSum = 0;
	  hDocumentSum = 0;
	  for (int i = 0; i < training.length; i++)
		  if (training[i].label.equals(Label.TRAGEDY)){
			  tDocumentSum++;
			  tTokenSum+= training[i].words.length;
			  for (int j = 0; j < training[i].words.length; j++){
				  vocabulary.add(training[i].words[j]);
				  if (!tragedyCounts.containsKey(training[i].words[j]))
					  tragedyCounts.put(training[i].words[j], 1);
				  else tragedyCounts.put(training[i].words[j], tragedyCounts.get(training[i].words[j])+1);
			  }
		  }
		  else if (training[i].label.equals(Label.COMEDY)){
			  cDocumentSum++;
			  cTokenSum+= training[i].words.length;
			  for (int j = 0; j < training[i].words.length; j++){
				  vocabulary.add(training[i].words[j]);
				  if (!comedyCounts.containsKey(training[i].words[j]))
					  comedyCounts.put(training[i].words[j], 1);
				  else comedyCounts.put(training[i].words[j], comedyCounts.get(training[i].words[j])+1);
			  }
		  }
		  else if (training[i].label.equals(Label.HISTORY)){
			  hDocumentSum++;
			  hTokenSum+= training[i].words.length;
			  for (int j = 0; j < training[i].words.length; j++){
				  vocabulary.add(training[i].words[j]);
				  if (!historyCounts.containsKey(training[i].words[j]))
					  historyCounts.put(training[i].words[j], 1);
				  else historyCounts.put(training[i].words[j], historyCounts.get(training[i].words[j])+1);
			  }
		  }
  }

  /*
   * Prints out the number of documents for each label
   * A sanity check method
   */
  public void documents_per_label_count(){
    System.out.println("TRAGEDY=" + tDocumentSum);
	System.out.println("COMEDY=" + cDocumentSum);
	System.out.println("HISTORY=" + hDocumentSum);
  }

  /*
   * Prints out the number of words for each label
	Another sanity check method
   */
  public void words_per_label_count(){
	  System.out.println("TRAGEDY=" + tTokenSum);
	  System.out.println("COMEDY=" + cTokenSum);
	  System.out.println("HISTORY=" + hTokenSum);
  }

  /**
   * Returns the prior probability of the label parameter, i.e. P(COMEDY) or P(TRAGEDY)
   */
  @Override
  public double p_l(Label label) {
	  double sum = 0.00001;
	  if (label.equals(Label.COMEDY))
		  sum = cDocumentSum*1.0;
	  else if (label.equals(Label.HISTORY))
		  sum = hDocumentSum*1.0;
	  else if (label.equals(Label.TRAGEDY))
		  sum = tDocumentSum*1.0;
    return sum/training.length;
  }

  /**
   * Returns the smoothed conditional probability of the word given the label, i.e. P(word|COMEDY) or
   * P(word|HISTORY)
   */
  @Override
  public double p_w_given_l(String word, Label label) {
    double sum1 = 0.00001;
    double sum2 = 0.00001;
    if (label.equals(Label.COMEDY)){
    	if (comedyCounts.get(word) != null)
    		sum1 = comedyCounts.get(word)*1.0;
    	else sum1 = 0.0;
    	sum2 = cTokenSum*1.0;
    }
    else if (label.equals(Label.HISTORY)){
    	if (historyCounts.get(word) != null)
    		sum1 = historyCounts.get(word)*1.0;
    	else sum1 = 0.0;
    	sum2 = hTokenSum*1.0;
    }
    else if (label.equals(Label.TRAGEDY)){
    	if (tragedyCounts.get(word) != null)
    		sum1 = tragedyCounts.get(word)*1.0;
    	else sum1 = 0.0;
    	sum2 = tTokenSum*1.0;
    }
    return (sum1 + 0.00001)/(0.00001*vocabulary.size() + sum2);
  }

  /**
   * Classifies a document as either a Comedy, History, or Tragedy.
   Break ties in favor of labels with higher prior probabilities.
   */
  @Override
  public Label classify(Instance ins) {
	  double tP = Math.log(p_l(Label.TRAGEDY));
	  double cP = Math.log(p_l(Label.COMEDY));
	  double hP = Math.log(p_l(Label.HISTORY));
	  for (int i = 0; i < ins.words.length; i++){
		  tP+= Math.log(p_w_given_l(ins.words[i], Label.TRAGEDY));
		  cP+= Math.log(p_w_given_l(ins.words[i], Label.COMEDY));
		  hP+= Math.log(p_w_given_l(ins.words[i], Label.HISTORY));
	  }
	  if (tP >= cP && tP >= hP)
		  return Label.TRAGEDY;
	  else if (cP >= tP && cP >= hP)
		  return Label.COMEDY;
	  else if (hP >= cP && hP >= tP)
		  return Label.HISTORY;
	  return null;
	
	//Initialize sum probabilities for each label
	//For each word w in document ins
		//compute the log (base e or default java log) probability of w|label for all labels (COMEDY, TRAGEDY, HISTORY)
		//add to appropriate sum
	//Return the Label of the maximal sum probability
  }
  
  
}
