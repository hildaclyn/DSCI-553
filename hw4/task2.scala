import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import java.io.{File, PrintWriter}
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, Map, Queue, Set}//, Set => MutableSet}


object task2 {
  def GNA(group: mutable.Map[String, mutable.Set[String]], nodes: Array[String]): Array[((String, String), Double)] = {
    val b_v = mutable.Map.empty[(String, String), Double]

    nodes.foreach { root =>
      val parent = mutable.Map.empty[String, mutable.Set[String]]
      val level = mutable.Map(root -> 0)
      val q = Queue[String](root)
      val visit = mutable.Set(root)
      val n_stp = mutable.Map(root -> 1.0)
      val path = ArrayBuffer(root)

      while (q.nonEmpty) {
        val current = q.dequeue()
        group.getOrElse(current, Set.empty).foreach { nb =>
          if (!visit.contains(nb)) {
            level += nb -> (level(current) + 1)
            visit += nb
            q.enqueue(nb)
            path += nb
            parent.getOrElseUpdate(nb, mutable.Set.empty) += current
            n_stp += nb -> (n_stp.getOrElse(nb, 0.0) + n_stp(current))
          } else if (level(nb) == level(current) + 1) {
            parent(nb) += current
            n_stp(nb) += n_stp(current)
          }
        }
      }

      val n_w = mutable.Map.empty[String, Double].withDefaultValue(1.0)
      val e_w = mutable.Map.empty[(String, String), Double].withDefaultValue(0.0)

      path.reverseIterator.foreach { rv =>
        parent.getOrElse(rv, Set.empty).foreach { b =>
          val w = n_w(rv) * (n_stp(b) / n_stp(rv))
          n_w(b) += w
          val edge_n = if (rv < b) (rv, b) else (b, rv)
          e_w(edge_n) += w
        }
      }

      e_w.foreach { case (key, value) =>
        b_v(key) = b_v.getOrElse(key, 0.0) + value / 2
      }
    }

    implicit val ordering: Ordering[((String, String), Double)] =
      Ordering.by[((String, String), Double), (Double, (String, String))](
        x => (-x._2, x._1)
      )
    b_v.toArray.sorted(ordering)
  }

  def calMod(
              comits: Array[Array[String]], // Assuming communities is an array of array of strings (node IDs)
              edgesG: Map[String, Set[String]], // Edges from a node to its set of neighbors
              m: Int, // Total number of edges
              kN: Map[String, Int] // Degree of each node
            ): Double = {
    var modularity = 0.0
    for (community <- comits) { // Iterate over each community
      for (i <- community) { // Iterate over each node i in the community
        for (j <- community) { // Iterate over each node j in the community
          val aIj = if (edgesG.getOrElse(i, Set.empty).contains(j)) 1.0 else 0.0
          modularity += (aIj - (kN(i) * kN(j)).toDouble / (2.0 * m))
        }
      }
    }
    modularity / (2.0 * m)
  }
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Task2")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    if (args.length != 4) {
      println("Usage: spark-submit --class Task2 <jar file> <threshold> <input_file> <betweenness_output> <community_output>")
      System.exit(1)
    }

    val threshold = args(0).toInt
    val inputPath = args(1)
    val betweenOutput = args(2)
    val communityOutput = args(3)

    val start = System.currentTimeMillis()

    val rawData = sc.textFile(inputPath)
    val header = rawData.first()
    val data = rawData.filter(_ != header)
      .map(_.split(","))
      .map(parts => (parts(0), Set(parts(1))))
      .reduceByKey(_ union _)

    // Filter users with at least `threshold` products, and prepare data structures for graph construction
    val data_pre = data.filter(_._2.size >= threshold).collectAsMap()
    val uni_user = data_pre.keys.toList
    val combuser = uni_user.combinations(2).toList// All possible pairs of users

    // Filter pairs of users with at least `threshold` shared products, and construct the graph
    val filter_p = sc.parallelize(combuser).filter(users => data_pre(users(0)).intersect(data_pre(users(1))).size >= threshold)
    val edges_n = filter_p.flatMap(users => Seq((users(1), users(0)), (users(0), users(1))))
    val nodes = filter_p.flatMap(users => Seq(users(0), users(1))).distinct()
    val n_list_a: List[String] = nodes.collect().toList

    val edges_kv = edges_n.map { case (user, friend) => (user, Set(friend)) }
    val edges_gr = edges_kv.reduceByKey(_ union _)
    val edges_g_e = edges_gr.collectAsMap()
    val edges_g: mutable.Map[String, mutable.Set[String]] = edges_g_e.map {
      case (key, immutableSet) => key -> (mutable.Set.empty[String] ++= immutableSet)
    }(collection.breakOut)
    val n_list = n_list_a.toArray

    var betweens: Array[((String, String), Double)] = GNA(edges_g, n_list)
    val output_1 = betweens.map { case (pair, between) =>
      (pair, BigDecimal(between).setScale(5, BigDecimal.RoundingMode.HALF_UP).toDouble)
    }
    val pw = new PrintWriter(new File(betweenOutput))
    output_1.map { case ((user1, user2), score) =>
      s"('$user1', '$user2'),$score"
    }.foreach(pw.println)
    pw.close()

    val m = betweens.length
    val l_g: RDD[(String, Int)] = edges_gr.map { case (node, edges) => (node, edges.size) }
    val k_n: mutable.Map[String, Int] = mutable.Map(l_g.collectAsMap().toSeq: _*)
    var max_mod = Double.NegativeInfinity
    var final_com: List[List[String]] = List()
    //var lo = betweens
    while (betweens.nonEmpty) {
      val ((node1, node2), _) = betweens.head
      // removes the selected edge from the graph to potentially fragment the network into communities
      edges_g.get(node1).foreach(_.remove(node2))
      edges_g.get(node2).foreach(_.remove(node1))

      var comits = ArrayBuffer[ArrayBuffer[String]]()
      var oriN = mutable.Set(n_list: _*)

      while (oriN.nonEmpty) {
        val root = oriN.head
        oriN.remove(root)
        val queue = Queue[String](root)
        val visitAg = mutable.Set[String](root)
        val com = ArrayBuffer[String](root)

        while (queue.nonEmpty) {
          val current = queue.dequeue()
          edges_g.getOrElse(current, mutable.Set.empty).foreach { nb =>
            if (!visitAg.contains(nb)) {
              visitAg += nb
              oriN -= nb
              queue.enqueue(nb)
              com += nb
            }
          }
        }
        comits += com.sorted
      }
      val curMod = calMod(comits.map(_.toArray).toArray, edges_g, m, k_n)

      if (curMod > max_mod) {
        max_mod = curMod
        //println(max_mod)
        final_com = comits.map(_.toList).toList
      }

      betweens = GNA(edges_g, n_list) // Recalculate betweenness
    }
    val result = final_com.sortBy(comm => (comm.size, comm.head))
    //result.foreach(println)
    try {
      val writer = new PrintWriter(new File(communityOutput))
      try {
        result.foreach(f => {
          val com = f.map(u => s"'$u'").mkString(", ")
          writer.write(com + "\n")
        })
      } finally {
        writer.close()
      }
    } catch {
      case e: Exception =>
        e.printStackTrace() // Print any exception to understand the failure
    }


    sc.stop()

    val end_time = System.currentTimeMillis()
    val execution = (end_time - start) / 1000.0
    println(s"Duration: $execution")
  }
}

