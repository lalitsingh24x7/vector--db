from pymilvus import connections, utility


def connect_to_milvus(host: str = "localhost", port: str = "19530") -> bool:
    """
    Connect to Milvus server.
    
    Args:
        host: Milvus server host
        port: Milvus server port
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
        print(f"Connected to Milvus at {host}:{port}")
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False


def disconnect_from_milvus(alias: str = "default") -> None:
    """Disconnect from Milvus server."""
    connections.disconnect(alias)
    print("Disconnected from Milvus")


def check_connection() -> bool:
    """Check if connected to Milvus and print server info."""
    try:
        server_version = utility.get_server_version()
        print(f"Milvus server version: {server_version}")
        return True
    except Exception as e:
        print(f"Connection check failed: {e}")
        return False


if __name__ == "__main__":
    if connect_to_milvus():
        check_connection()
        disconnect_from_milvus()
