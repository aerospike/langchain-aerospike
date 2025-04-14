Usage
=====

The ``Aerospike`` vector store wraps the ``aerospike_vector_search`` client to provide a langchain compatible vector store interface.
Indexes must be created with the ``aerospike_vector_search`` client before using the ``Aerospike`` vector store.
Target indexes with the ``Aerospike`` ``name`` parameter.

Below is an example of writing data using the vector store to an index that already exists.
The examples afterwards show the complete process of creating an index and vector store from scratch.

Basic Usage
-----------

Here is a basic example of creating a vector store and adding documents to it.

.. code-block:: python

    from langchain_aerospike.vectorstores import Aerospike
    from langchain_huggingface import HuggingFaceEmbeddings
    from aerospike_vector_search import Client, HostPort, types

    INDEX_NAME = "example-index"
    NAMESPACE = "test"
    VECTOR_FIELD = "vector"
    DIMENSIONS = 384  # This matches the model's output dimension
    DISTANCE_METRIC = types.VectorDistanceMetric.COSINE

    client = Client(seeds=[HostPort(host="localhost", port=5000)], is_loadbalancer=True)

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
    )

    vector_store = Aerospike(
        client=client,
        embedding=embedding_model,
        namespace=NAMESPACE,
        index_name=INDEX_NAME,
        vector_key=VECTOR_FIELD,
    )

    # Add documents to the vector store
    texts = [
        "Aerospike is a real-time, distributed NoSQL database and vector database",
        "Vector databases store and retrieve vector embeddings for AI applications",
        "LangChain is a framework for developing applications powered by language models",
    ]
    metadatas = [
        {"source": "aerospike", "category": "database"},
        {"source": "vector_db", "category": "database"},
        {"source": "langchain", "category": "framework"},
    ]

    document_ids = vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
    )
    print(f"Added {len(document_ids)} documents to Aerospike")
    
    # Clean up resources
    try:
        # Delete the documents
        vector_store.delete(ids=document_ids)
        # Delete the index
        client.index_drop(namespace=NAMESPACE, name=INDEX_NAME)
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        # Close the client
        client.close()
    
    
Using a Standalone Index
------------------------

Here is a complete example of creating an index and a vector store, adding documents to the vector store, and searching for similar documents.

.. note::
   This example uses a standalone index so that the example data is indexed at index creation time.
   Your Aerospike Vector Search cluster must be configured with the ``standalone-indexer`` node role on at least 1 node for the following example to work.
   See the `Aerospike Vector Search documentation <https://aerospike.com/docs/vector/manage/config/#node-roles>`_ for more information.


.. code-block:: python

    from langchain_aerospike.vectorstores import Aerospike
    from langchain_huggingface import HuggingFaceEmbeddings
    from aerospike_vector_search import Client, HostPort, types

    INDEX_NAME = "example-index"
    NAMESPACE = "test"
    VECTOR_FIELD = "vector"
    DIMENSIONS = 384  # This matches the model's output dimension
    DISTANCE_METRIC = types.VectorDistanceMetric.COSINE


    def wait_for_index_ready(client, namespace, index_name, timeout=30) -> None:
        """Wait until the index is ready for search."""

        import time
        while True:
            index_status = client.index_get_status(namespace=namespace, name=index_name)
            if index_status.readiness == types.IndexReadiness.READY:
                break
            time.sleep(0.25)
            timeout -= 0.25
            if timeout <= 0:
                raise Exception("timed out waiting for index to become ready, "
                                "maybe standalone indexing is not configured on this AVS cluster")


    # Initialize the Aerospike client
    # using a load balancer with AVS is best practice
    # so is_loadbalancer is set True here
    # you should set this to False if you are not using a load balancer with an AVS cluster of more than 1 node
    client = Client(seeds=[HostPort(host="localhost", port=5000)], is_loadbalancer=True)

    # Initialize the embeddings model
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # A small, efficient embedding model
        model_kwargs={'device': 'cpu'},  # Use CPU for inference
    )

    # Create an Aerospike vector store
    vector_store = Aerospike(
        client=client,
        embedding=embedding_model,
        namespace=NAMESPACE,
        index_name=INDEX_NAME,
        vector_key=VECTOR_FIELD,
    )

    try:
        # Add documents to the vector store
        texts = [
            "Aerospike is a real-time, distributed NoSQL database and vector database",
            "Vector databases store and retrieve vector embeddings for AI applications",
            "LangChain is a framework for developing applications powered by language models",
        ]
        metadatas = [
            {"source": "aerospike", "category": "database"},
            {"source": "vector_db", "category": "database"},
            {"source": "langchain", "category": "framework"},
        ]

        # Add texts to the vector store
        document_ids = vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
        )
        print(f"Added {len(document_ids)} documents to Aerospike")

        # Create an index in AVS - for standalone mode, create after adding documents
        client.index_create(
            namespace=NAMESPACE,
            name=INDEX_NAME,
            vector_field=VECTOR_FIELD,
            dimensions=DIMENSIONS,
            mode=types.IndexMode.STANDALONE,
            vector_distance_metric=DISTANCE_METRIC,
        )

        # Wait for the index to be ready
        wait_for_index_ready(client, NAMESPACE, INDEX_NAME)

        # Search for similar documents
        query = "Tell me about vector databases"
        docs = vector_store.similarity_search(query, k=2)

        # Print the results
        print("\nSearch Results:")
        for i, doc in enumerate(docs):
            print(f"Result {i+1}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
    
    finally:
        # Clean up resources
        try:
            # Delete the documents
            vector_store.delete(ids=document_ids)
            # Delete the index
            client.index_drop(namespace=NAMESPACE, name=INDEX_NAME)
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            # Close the client
            client.close() 

Using a Distributed Index
--------------------------

Here is an example using a distributed index instead of a standalone index. In this pattern, you should create the index before inserting the documents, and the default distributed indexing service will take care of indexing the documents as they're inserted.

.. code-block:: python

    from langchain_aerospike.vectorstores import Aerospike
    from langchain_huggingface import HuggingFaceEmbeddings
    from aerospike_vector_search import Client, HostPort, types

    INDEX_NAME = "distributed-index-example"
    NAMESPACE = "test"
    VECTOR_FIELD = "vector"
    DIMENSIONS = 384  # This matches the model's output dimension
    DISTANCE_METRIC = types.VectorDistanceMetric.COSINE

    # Initialize the Aerospike client
    client = Client(seeds=[HostPort(host="localhost", port=5000)], is_loadbalancer=True)

    # First, create the index with DISTRIBUTED mode (this is the default)
    client.index_create(
        namespace=NAMESPACE,
        name=INDEX_NAME,
        vector_field=VECTOR_FIELD,
        dimensions=DIMENSIONS,
        mode=types.IndexMode.DISTRIBUTED,  # This is the default mode
        vector_distance_metric=DISTANCE_METRIC,
    )

    # Initialize the embeddings model
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
    )

    # Create an Aerospike vector store
    vector_store = Aerospike(
        client=client,
        embedding=embedding_model,
        namespace=NAMESPACE,
        index_name=INDEX_NAME,
        vector_key=VECTOR_FIELD,
    )

    try:
        # Add documents to the vector store
        texts = [
            "Aerospike is a real-time, distributed NoSQL database and vector database",
            "Vector databases store and retrieve vector embeddings for AI applications",
            "LangChain is a framework for developing applications powered by language models",
        ]
        metadatas = [
            {"source": "aerospike", "category": "database"},
            {"source": "vector_db", "category": "database"},
            {"source": "langchain", "category": "framework"},
        ]

        # Add texts to the vector store - with distributed indexing, documents are indexed as they're inserted
        document_ids = vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            # No need to wait for index, as it's handled asynchronously
            wait_for_index=False,
        )
        print(f"Added {len(document_ids)} documents to Aerospike")

        # Note: With distributed indexing, there may be a short delay before newly added 
        # documents appear in search results. For production applications, you might want
        # to monitor the index status using:
        #
        # percent_unmerged = client.index_get_percent_unmerged(
        #     namespace=NAMESPACE, 
        #     name=INDEX_NAME
        # )
        # print(f"Unmerged records: {percent_unmerged}%")
        
        # For this example, we'll add a small delay to allow for indexing
        import time
        time.sleep(1)

        # Search for similar documents
        query = "Tell me about vector databases"
        docs = vector_store.similarity_search(query, k=2)

        # Print the results
        print("\nSearch Results:")
        for i, doc in enumerate(docs):
            print(f"Result {i+1}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
    
    finally:
        # Clean up resources
        try:
            # Delete the documents
            vector_store.delete(ids=document_ids)
            # Delete the index
            client.index_drop(namespace=NAMESPACE, name=INDEX_NAME)
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            # Close the client
            client.close() 