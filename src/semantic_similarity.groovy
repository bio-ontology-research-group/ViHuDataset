@Grab(group='com.github.sharispe', module='slib-sml', version='0.9.1')
@Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.2.5')
@Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.2.5')
@Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.2.5')
@Grab(group='ch.qos.logback', module='logback-classic', version='1.2.3')
@Grab(group='org.slf4j', module='slf4j-api', version='1.7.30')
@Grab(group='org.codehaus.gpars', module='gpars', version='1.1.0')
@Grab('me.tongfei:progressbar:0.9.3')


import org.semanticweb.owlapi.model.*
import  org.semanticweb.owlapi.apibinding.OWLManager

import slib.sml.sm.core.engine.SM_Engine
import slib.sml.sm.core.measures.Measure_Groupwise
import slib.sml.sm.core.metrics.ic.utils.*
import slib.sml.sm.core.utils.SMConstants
import slib.utils.ex.SLIB_Exception
import slib.sml.sm.core.utils.SMconf
import slib.graph.model.impl.graph.memory.GraphMemory
import slib.graph.io.conf.GDataConf
import slib.graph.io.util.GFormat
import slib.graph.io.loader.GraphLoaderGeneric
import slib.graph.model.impl.repo.URIFactoryMemory
import slib.graph.model.impl.graph.elements.*
import slib.sml.sm.core.metrics.ic.utils.*
import slib.graph.algo.utils.*

import org.openrdf.model.vocabulary.RDF

import groovyx.gpars.GParsPool

import java.util.HashSet

import groovy.cli.commons.CliBuilder
import java.nio.file.Paths


import org.slf4j.Logger
import org.slf4j.LoggerFactory

// Initialize the logger

import java.util.logging.Logger
import java.util.logging.Level
import java.util.logging.ConsoleHandler
import java.util.logging.SimpleFormatter

// Initialize the logger
Logger logger = Logger.getLogger(this.class.name)

// Configure the logger
ConsoleHandler handler = new ConsoleHandler()
handler.setLevel(Level.ALL)
handler.setFormatter(new SimpleFormatter())
logger.addHandler(handler)
logger.setLevel(Level.ALL)
logger.setUseParentHandlers(false)




def cli = new CliBuilder(usage: 'semantic_similarity.groovy -r <root_dir> -ic <ic_measure> -pw <pairwise_measure> -gw <groupwise_measure>')
cli.r(longOpt: 'root_dir', args: 1, defaultValue: '../data', 'Root directory')
cli.ic(longOpt: 'ic_measure', args: 1, defaultValue: 'resnik', 'Information Content measure')
cli.pw(longOpt: 'pairwise_measure', args: 1, defaultValue: 'resnik', 'Pairwise measure')
cli.gw(longOpt: 'groupwise_measure', args: 1, defaultValue: 'bma', 'Groupwise measure')

def options = cli.parse(args)
if (!options) return

String rootDir = options.r
String icMeasure = options.ic
String pairwiseMeasure = options.pw
String groupwiseMeasure = options.gw

def manager = OWLManager.createOWLOntologyManager()
def ontology = manager.loadOntologyFromOntologyDocument(new File(rootDir + "/train.owl"))

def classes = ontology.getClassesInSignature().collect { it.toStringID() }

def evalGenes = classes.findAll {
    def parts = it.split("/")
    def lastPart = parts[-1]
    it.contains("mowl.borg")
}.sort()

println evalGenes.take(5)

logger.info("Total evaluation genes: ${evalGenes.size()}")

def existingMpPhenotypes = new HashSet()
def existingHpPhenotypes = new HashSet()
classes.each { cls ->
    if (cls.contains("MP_")) {
        existingMpPhenotypes.add(cls)
    } else if (cls.contains("HP_")) {
        existingHpPhenotypes.add(cls)
    }
}



logger.info("Obtaining Gene-Phenotype associations from gene_to_phenotype.csv. Genes are represented as 'http://mowl.borg/[entrez] and Phenotypes are represented as MP IDs")
def gene2pheno = new HashMap()

def gene_to_phenotype_file = new File(rootDir + "/gene_to_phenotype.csv")
def gene_to_phenotype = gene_to_phenotype_file.readLines()*.split(',')
gene_to_phenotype.each { line ->
    def gene = line[0]
    def phenotype = line[1]
    
    if (phenotype in existingHpPhenotypes || phenotype in existingMpPhenotypes) {
	if (!gene2pheno.containsKey(gene)) {
	    gene2pheno[gene] = new HashSet()
	}
        gene2pheno[gene].add(phenotype)    
    }
}

logger.info("gene2pheno size: ${gene2pheno.size()}")
logger.info("E.g. gene2pheno: ${gene2pheno.take(1)}") 



def test_ontology = manager.loadOntologyFromOntologyDocument(new File(rootDir + "/test.owl"))

def testPairs = axiomsToPairs(test_ontology, logger)

logger.info("Total test pairs: ${testPairs.size()}")
logger.info("E.g. test pair: ${testPairs.take(1)}")


logger.info("Obtaining Virus-Phenotype associations from virus_to_phenotype.csv. Virus are represented as 'http://purl.obolibrary.org/obo/NCBITaxon_[taxon id]' and Phenotypes are represented as HP IDs")
def virus_pheno_file = new File(rootDir + "/virus_to_phenotype.csv")
def virus_to_pheno = virus_pheno_file.readLines()*.split(',')

def virus2pheno = new HashMap()

virus_to_pheno.each { line ->
    def parts = line
    def virus = parts[0]
    def phenotype = parts[1]

    if (existingHpPhenotypes.contains(phenotype)) {
	if (! virus2pheno.containsKey(virus)) {
	    virus2pheno[virus] = []
	}
	virus2pheno[virus].add(phenotype)
    }
}

logger.info("virus2pheno size: ${virus2pheno.size()}")
logger.info("E.g. virus2pheno: ${virus2pheno.take(1)}")

logger.info("Preparing Semantic Similarity Engine")
def factory = URIFactoryMemory.getSingleton()
def graphUri = factory.getURI("http://purl.obolibrary.org/obo/HPI_")
factory.loadNamespacePrefix("HPI", graphUri.toString())
def graph = new GraphMemory(graphUri)
def goConf = new GDataConf(GFormat.RDF_XML, Paths.get(rootDir, "phenomenet-inferred.owl").toString())

GraphLoaderGeneric.populate(goConf, graph)

def virtualRoot = factory.getURI("http://purl.obolibrary.org/obo/HPI_virtual_root")
def rooting = new GAction(GActionType.REROOTING)
rooting.addParameter("root_uri", virtualRoot.stringValue())
GraphActionExecutor.applyAction(factory, rooting, graph)



def withAnnotations = true

if (withAnnotations) {
    gene2pheno.each { gene, phenotypes ->
	phenotypes.each { phenotype ->
            def geneId = factory.getURI(gene)
	    def phenotypeId = factory.getURI(phenotype)

	    if (!graph.containsVertex(phenotypeId)) {
		throw new Exception("Graph does not contain gene vertex: ${phenotypeId}")
	    }
            
	    Edge e = new Edge(geneId, RDF.TYPE, phenotypeId)
	    graph.addE(e)
	}
    }

    // virus2pheno.each { virus, phenotypes ->
	// phenotypes.each { phenotype ->
	    // def virusId = factory.getURI(virus)
	    // def phenotypeId = factory.getURI(phenotype)
	    // Edge e = new Edge(virusId, RDF.TYPE, phenotypeId)
	    // graph.addE(e)

	// }
        
    // }
}



def engine = new SM_Engine(graph)

def icConf = null

if (withAnnotations) {
    icConf = new IC_Conf_Corpus(icMeasureResolver(icMeasure))
}
else {
    icConf = new IC_Conf_Topo(icMeasureResolver(icMeasure))
}
    

def smConfPairwise = new SMconf(pairwiseMeasureResolver(pairwiseMeasure))
smConfPairwise.setICconf(icConf)
def smConfGroupwise = new SMconf(groupwiseMeasureResolver(groupwiseMeasure))

def mr = 0
def mrr = 0
def hitsK = [1: 0, 3: 0, 10: 0, 100: 0]
def ranks = [:]



logger.info("Computing Semantic Similarity for ${testPairs.size()} Virus-Human pairs")
logger.info("Starting Pool ")
 
    def allRanks = GParsPool.withPool {
    testPairs.collectParallel { pair ->

	def test_gene = pair[0]
	def test_virus = pair[1]
	def virus_phenotypes = virus2pheno[test_virus].collect { factory.getURI(it) }.toSet()
	if (virus_phenotypes.size() == 0) {
	    logger.warning("No phenotypes for virus: ${test_virus}. Setting scores to zero.")
	}
	
	try {
	    scores = evalGenes.collect { gene ->
		def phenotypes = gene2pheno[gene]
		def gene_phenotypes = phenotypes.collect { factory.getURI(it) }.toSet()

		if (gene_phenotypes.size() == 0) {
                    logger.warning("No phenotypes for gene: ${gene}. Setting scores to zero.")
		}
		
		def sim_score = engine.compare(smConfGroupwise, smConfPairwise, gene_phenotypes, virus_phenotypes)
		sim_score
	    }

	}catch (Exception e) {
	    println "Error computing similarity for pair: ${pair}. Setting scores to zero. Error: ${e.message}"
	    println "Phenotypes for gene: ${gene2pheno[pair[0]]}"
	    println "Phenotypes for virus: ${virus2pheno[pair[1]]}"
            throw e
	    
	}
	
	def test_gene_index = evalGenes.indexOf(test_gene)
	
	[test_gene, test_virus, test_gene_index, scores]
    }

}


def out_file = rootDir + "/results_${icMeasure}_${pairwiseMeasure}_${groupwiseMeasure}.txt"
def out = new File(out_file)
out.withWriter { writer ->
	allRanks.each { r ->
	def gene = r[0]
	def virus = r[1]
	def gene_index = r[2]
	def scores = r[3]
	writer.write("${gene}\t${virus}\t${gene_index}\t${scores.join("\t")}\n")
	}
}


logger.info("Done")
// out.close()


logger.info("Results written to ${rootDir}/results.txt")



// logger.info("Pool finished. Analyzing results")


def axiomsToPairs(ontology, logger) {
    def pairs = []
    ontology.getAxioms().each { axiom ->
        if (axiom.getAxiomType() == AxiomType.SUBCLASS_OF) {
            def superclass = axiom.getSuperClass()
            if (superclass.getClassExpressionType() == ClassExpressionType.OBJECT_SOME_VALUES_FROM) {
                def subclass = axiom.getSubClass()
                def prop = superclass.getProperty()
                def filler = superclass.getFiller()
		
                def gene = subclass.toStringID()
                def virus = filler.toStringID()

		if (prop.toStringID() == "http://mowl.borg/associated_with") {
		    pairs.add([gene, virus])
                    
		}
		    
            }
        }
    }

    logger.info("Total evaluation pairs: ${pairs.size()}")
    return pairs
}

static computeRankRoc(ranks, numEntities) {
    def nTails = numEntities

    def aucX = ranks.keySet().sort()
    def aucY = []
    def tpr = 0
    def sumRank = ranks.values().sum()
    aucX.each { x ->
        tpr += ranks[x]
        aucY.add(tpr / sumRank)
    }
    aucX.add(nTails)
    aucY.add(1)
    def auc = 0
        for (int i = 1; i < aucX.size(); i++) {
        auc += (aucX[i] - aucX[i-1]) * (aucY[i] + aucY[i-1]) / 2
    }
    return auc / nTails
}

static icMeasureResolver(measure) {
    if (measure.toLowerCase() == "sanchez") {
        return SMConstants.FLAG_ICI_SANCHEZ_2011
    } else if (measure.toLowerCase() == "resnik") {
	return SMConstants.FLAG_IC_ANNOT_RESNIK_1995_NORMALIZED

    } else {
        throw new IllegalArgumentException("Invalid IC measure: $measure")
    }
}

static pairwiseMeasureResolver(measure) {
    if (measure.toLowerCase() == "lin") {
        return SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_LIN_1998
    } else if (measure.toLowerCase() == "resnik") {
	return SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_RESNIK_1995
    } else {
        throw new IllegalArgumentException("Invalid pairwise measure: $measure")
    }
}

static groupwiseMeasureResolver(measure) {
    if (measure.toLowerCase() == "bma") {
        return SMConstants.FLAG_SIM_GROUPWISE_BMA
    } else if (measure.toLowerCase() == "bmm") {
	return SMConstants.FLAG_SIM_GROUPWISE_BMM
    } else {
        throw new IllegalArgumentException("Invalid groupwise measure: $measure")
    }
}

