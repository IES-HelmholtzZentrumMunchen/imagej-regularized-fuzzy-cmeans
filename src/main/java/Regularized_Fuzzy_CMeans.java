import java.util.Arrays;
import java.util.Random;
import java.util.Vector;

import ij.*;
import ij.gui.GenericDialog;
import ij.gui.NewImage;
import ij.plugin.filter.PlugInFilter;
import ij.process.*;


/**
 * Fuzzy c-means clustering.
 *
 * A clustering algorithm using Modified FCM.
 *
 *
 * @author Julien Pontabry
 */
public class Regularized_Fuzzy_CMeans implements PlugInFilter {
    /**
     * Number of classes to classify.
     */
	private int m_numberOfClasses = 2;

    /**
     * Maximal number of iterations of algorithm.
     */
	private int m_maxNumberOfIterations = 100;

    /**
     * Stopping criterion of the algorithm.
     */
	private double m_epsilon = 1e-4;

    /**
     * Regularization factor of the algorithm.
     */
	private double m_regularization = 20;

    /**
     * Number of pixels in neighborhood.
     */
	private int m_numberOfNeighbors = 24;

    /**
     * Patch size defining the neighborhood.
     */
    private int m_patchSize = 5;

    /**
     * Input image to classify.
     */
    private ImagePlus m_inputImage;
	
	/**
	 * Setup method
	 * @param arg Arguments of the plugin.
	 * @param imp Image on which the plugin will be processed.
	 */
	public int setup(String arg, ImagePlus imp)
	{
		// Create the GUI
		GenericDialog window = new GenericDialog("Fuzzy C-means parameters");
		window.addNumericField("Number of classes", m_numberOfClasses, 0);
		window.addNumericField("Max number of iterations", m_maxNumberOfIterations, 0);
		window.addNumericField("Convergence threshold", m_epsilon, 4);
		window.addNumericField("Regularization", m_regularization, 3);
		window.addNumericField("Neighborhood", m_patchSize, 0);
		window.showDialog();
		
		
		// Checking GUI events
		if(window.wasCanceled())
		{
			return DONE;
		}
			
		if(window.invalidNumber())
		{
			IJ.error("Fuzzy C-means clustering error", "A numerical field is not filled correctly !");
			return DONE;
		}

		// Get back the parameters from the GUI
		m_numberOfClasses       = (int)window.getNextNumber();  // The first parameter is the number of classes
		m_maxNumberOfIterations = (int)window.getNextNumber();  // The second parameter is the max number of iterations
		m_epsilon               = window.getNextNumber();       // The third parameter is the convergence epsilon
		m_regularization        = window.getNextNumber();       // The regularization parameter
		m_patchSize             = (int)window.getNextNumber();  // The size of patch neighbors of a pixel used for regularization

        if(imp.getNSlices() > 1)
		    m_numberOfNeighbors = m_patchSize*m_patchSize*m_patchSize - 1;
        else // imp.getNSlices() == 1
            m_numberOfNeighbors = m_patchSize*m_patchSize - 1;
		
		
		// Checking parameters
		if(m_numberOfClasses <= 0)
		{
			IJ.error("Fuzzy C-means clustering error", "The number of classes should be greater than 0 !");
			return DONE;
		}
		
		if(m_maxNumberOfIterations <= 0)
		{
			IJ.error("Fuzzy C-means clustering error", "The maximal number of iterations should be greater than 0 !");
			return DONE;
		}

        m_inputImage = imp;

		return NO_CHANGES+DOES_8G+DOES_16+DOES_32;
	}
	
	/**
	 * Actually run the plugin processing.
	 * @param ip Image processor on which processing is done.
	 */
	public void run(ImageProcessor ip)
	{
        Vector< ImagePlus > clustering = this.classify(m_inputImage);

        for(ImagePlus classImage : clustering) {
            classImage.show();
        }
	}

    public Vector< ImagePlus > classify(ImagePlus input) {
        //////////////////////////////////////////////////////
        // Initialisation
        //////////////////////////////////////////////////////

        // Compute temporary image for regularisation
        ImagePlus tmp;

        if(input.getNSlices() > 1)
            tmp = this.computeTemporaryImage3D(input);
        else // input.getNSlices() == 1
            tmp = this.computeTemporaryImage2D(input);

        // Linearise tmp image
        double[] tmp_intensities = this.lineariseImage(tmp);

        // Initialise output data (labels for each pixel)
        Vector< double[] > labels_intensities = new Vector< double[] >();

        for(int currentMeanIndex = 0; currentMeanIndex < m_numberOfClasses; currentMeanIndex++) {
            labels_intensities.add(new double[tmp_intensities.length]);
        }

        // Get minimal, maximal and range intensities
        double minIntensity = 1000000000;
        double maxIntensity = 0;

        for(double currentIntensity : tmp_intensities) {
//            if(mask[currentPixelIndex] != 0) {
                if(currentIntensity > maxIntensity)
                    maxIntensity = currentIntensity;

                if(currentIntensity < minIntensity)
                    minIntensity = currentIntensity;
//            } // if is in mask
        } // for each pixel

        double intensityDif = maxIntensity - minIntensity;

        // Random initialization of the means
        double[] means = new double[m_numberOfClasses];

        Random randomiser = new Random();
        for(int currentMeanIndex = 0; currentMeanIndex < means.length; currentMeanIndex++)
        {
            means[currentMeanIndex] = randomiser.nextDouble() * intensityDif + minIntensity;
        } // for each class


        //////////////////////////////////////////////////////
        // Processing
        //////////////////////////////////////////////////////

        // Initialise local variables
        double[]    distances = new double[m_numberOfClasses];
//        double    convergence = 1.0;
        int currentLoopNumber = 0;

        while(currentLoopNumber < m_maxNumberOfIterations/* && convergence > m_epsilon*/) {
            // Compute the pixels' fuzzy labels depending on the distance
            // of their intensities and the means intensities
            for(int currentPixelIndex = 0; currentPixelIndex < tmp_intensities.length; currentPixelIndex++) {
//                if(mask[currentPixelIndex] != 0) {
                    // Compute distance to intensity value for each classes
                    double sumOfDistances = 0.0;

                    for(int currentMeanIndex = 0; currentMeanIndex < means.length; currentMeanIndex++) {
                        distances[currentMeanIndex]  = (tmp_intensities[currentPixelIndex] - means[currentMeanIndex]);
                        distances[currentMeanIndex] *= distances[currentMeanIndex];

                        sumOfDistances += distances[currentMeanIndex];
                    } // for each mean

                    // Compute the fuzzy coefficients of pixel for each class
                    for(int currentMeanIndex = 0; currentMeanIndex < means.length; currentMeanIndex++) {
                        labels_intensities.get(currentMeanIndex)[currentPixelIndex] = distances[currentMeanIndex] / sumOfDistances;
                    } // for each class
//                } // if pixel is in mask
            } // for each intensity value

            // Compute the new means with the fuzzy coefficients
            double[] sumOfFuzzyLabels = new double[m_numberOfClasses];
//            double[]       means_copy = means;
            means                     = new double[m_numberOfClasses];

            for(int currentPixelIndex = 0; currentPixelIndex < tmp_intensities.length; currentPixelIndex++) {
//                if(mask[currentPixelIndex] != 0) {
                    for(int currentMeanIndex = 0; currentMeanIndex < means.length; currentMeanIndex++) {
                        double fuzzyCoefficient = labels_intensities.get(currentMeanIndex)[currentPixelIndex];
                        means[currentMeanIndex] += fuzzyCoefficient * tmp_intensities[currentPixelIndex];
                        sumOfFuzzyLabels[currentMeanIndex] += fuzzyCoefficient;
                    } // for each class
//                } // if pixel is in mask
            } // for each pixel

            for(int currentMeanIndex = 0; currentMeanIndex < means.length; currentMeanIndex++)
            {
                means[currentMeanIndex] /= sumOfFuzzyLabels[currentMeanIndex];
            } // for each mean

//            // FIXME : strangely, the means are swapped and we are forced to compute a "global" distance
//            // Update loop information (convergence and number of iterations)
//            convergence = 0.0;
//
//            for(int currentMeanIndex = 0; currentMeanIndex < means.length; currentMeanIndex++)
//            {
////                convergence += Math.abs(means[currentMeanIndex] - means_copy[currentMeanIndex]);
//                convergence += means[currentMeanIndex];
//                convergence -= means_copy[currentMeanIndex];
//            } // for each mean

            currentLoopNumber++;
        }


        //////////////////////////////////////////////////////
        // Post-processing
        //////////////////////////////////////////////////////

        // Initialize output class images
        Vector< ImagePlus > clustering = new Vector< ImagePlus >();

        // Sort means array
        double[] means_sort = means.clone();
        Arrays.sort(means_sort);

        // Create each image for each class
        for(int currentMeanIndex = 0; currentMeanIndex < labels_intensities.size(); currentMeanIndex++) {
            // Find ranking
            int finalIndex = 0;

            for(int otherMeanIndex = 0; otherMeanIndex < means.length; otherMeanIndex++) {
                if(Double.compare(means[currentMeanIndex], means_sort[otherMeanIndex]) == 0)
                    finalIndex = otherMeanIndex;
            }

            // Create a new image by delinearisation)
            clustering.add(this.delineariseImage(labels_intensities.get(finalIndex), input.getWidth(), input.getHeight(), input.getNSlices(), 32, "Class " + String.valueOf(finalIndex)));
        }


        return clustering;
    }

    /**
     * Linearise an image.
     * @param input Input image.
     * @return Linearised image (double precision 1D-array).
     */
    private double[] lineariseImage(ImagePlus input) {
        // Initialise output
        double[] output = new double[input.getWidth()*input.getHeight()*input.getNSlices()];

        // Initalise linear index
        int linearIndex = 0;

        // Get input's stack
        ImageStack inputStack = input.getStack();

        // Go through each voxel (ZYX order)
        for(int z = 0; z < inputStack.getSize(); z++) {
            for(int y = 0; y < inputStack.getHeight(); y++) {
                for(int x = 0; x < inputStack.getWidth(); x++) {
                    output[linearIndex++] = inputStack.getVoxel(x,y,z);
                }
            }
        }

        return output;
    }

    /**
     * Delinearise an image.
     * @param input Input linearised image (double precision 1D-array).
     * @param width Width of the output image.
     * @param height Height of the output image.
     * @param depth Depth of the output image.
     * @param bitdepth Bit depth of the output image.
     * @param name Title of the image (window title).
     * @return The output image (ImageJ format).
     */
    private ImagePlus delineariseImage(double[] input, int width, int height, int depth, int bitdepth, String name) {
        // Initialise output
        ImagePlus output = NewImage.createImage(name, width, height, depth, bitdepth, NewImage.FILL_BLACK);

        // Initalise linear index
        int linearIndex = 0;

        // Get output's stack
        ImageStack outputStack = output.getStack();

        // Go through each voxel (ZYX order)
        for(int z = 0; z < outputStack.getSize(); z++) {
            for(int y = 0; y < outputStack.getHeight(); y++) {
                for(int x = 0; x < outputStack.getWidth(); x++) {
                    outputStack.setVoxel(x,y,z, input[linearIndex++]);
                }
            }
        }

        return output;
    }

    /**
     * Compute the temporary image for 3D input image.
     * @param input Input image.
     * @return The temporary image.
     */
    private ImagePlus computeTemporaryImage3D(ImagePlus input) {
        // Create a new temporary image
        ImagePlus tmp_image = NewImage.createFloatImage("Temporary image", input.getWidth(), input.getHeight(), input.getNSlices(), NewImage.FILL_BLACK);

        // Get the image stacks
        ImageStack inputStack = input.getStack();
        ImageStack  tmp_stack = tmp_image.getStack();

        // Compute coefficients
        double coefficient1 = 1.0 / (1.0 + m_regularization);
        double coefficient2 = m_regularization / m_numberOfNeighbors;

        // Compute offset due to patch size
        int  offset = (m_patchSize - 1) / 2;

        // Go through each pixel
        for(int z = offset; z < tmp_stack.getSize()-offset; z++) {
            for(int y = offset; y < tmp_stack.getHeight()-offset; y++) {
                for(int x = offset; x < tmp_stack.getWidth()-offset; x++)
                {
                    double sumOfIntensities = 0.0;

                    // Go through the neighborhood
                    for(int delta_z = -offset; delta_z <= offset; delta_z++) {
                        for(int delta_y = -offset; delta_y <= offset; delta_y++) {
                            for(int delta_x = -offset; delta_x <= offset; delta_x++) {
                                if(delta_x != x || delta_y != y || delta_z != z)
                                    sumOfIntensities += inputStack.getVoxel(x+delta_x, y+delta_y, z+delta_z);
                            } // for delta_x
                        } // for delta_y
                    } // for delta_z

                    tmp_stack.setVoxel(x,y,z, coefficient1 * (inputStack.getVoxel(x,y,z) + coefficient2 * sumOfIntensities));
                } // for x
            } // for y
        } // for z

        return tmp_image;
    }

    /**
     * Compute the temporary image for 2D input image.
     * @param input Input image.
     * @return The temporary image.
     */
    private ImagePlus computeTemporaryImage2D(ImagePlus input) {
        // Create a new temporary image
        ImagePlus tmp_image = NewImage.createFloatImage("Temporary image", input.getWidth(), input.getHeight(), 1, NewImage.FILL_BLACK);

        // Get the image stacks
        ImageStack inputStack = input.getStack();
        ImageStack  tmp_stack = tmp_image.getStack();

        // Compute coefficients
        double coefficient1 = 1.0 / (1.0 + m_regularization);
        double coefficient2 = m_regularization / m_numberOfNeighbors;

        // Compute offset due to patch size
        int  offset = (m_patchSize - 1) / 2;

        // Go through each pixel
        for(int y = offset; y < tmp_stack.getHeight()-offset; y++) {
            for(int x = offset; x < tmp_stack.getWidth()-offset; x++)
            {
                double sumOfIntensities = 0.0;

                // Go through the neighborhood
                for(int delta_y = -offset; delta_y <= offset; delta_y++) {
                    for(int delta_x = -offset; delta_x <= offset; delta_x++) {
                        if(delta_x != x || delta_y != y)
                            sumOfIntensities += inputStack.getVoxel(x+delta_x, y+delta_y, 0);
                    } // for delta_x
                } // for delta_y

                tmp_stack.setVoxel(x,y,0, coefficient1 * (inputStack.getVoxel(x,y,0) + coefficient2 * sumOfIntensities));
            } // for x
        } // for y

        return tmp_image;
    }
	
	/**
	 * Main method for debugging.
	 *
	 * For debugging, it is convenient to have a method that starts ImageJ, loads an
	 * image and calls the plugin, e.g. after setting breakpoints.
	 *
	 * @param args unused
	 */
	public static void main(String[] args) {
		// set the plugins.dir property to make the plugin appear in the Plugins menu
		Class<?> clazz = Regularized_Fuzzy_CMeans.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring(5, url.length() - clazz.getName().length() - 6);
		System.setProperty("plugins.dir", pluginsDir);

		// start ImageJ
		new ImageJ();
	}
}
