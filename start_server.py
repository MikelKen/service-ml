#!/usr/bin/env python3
"""
Minimal server startup test to validate GraphQL field name fixes
"""
import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("🚀 Starting clustering microservice...")
    print("📋 Testing GraphQL schema validation...")
    
    # Test import of core components
    from app.main import app
    print("✅ FastAPI app imported successfully")
    
    # Test GraphQL schema creation
    import strawberry
    from app.graphql.simple_ml import Query, Mutation
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    print("✅ GraphQL schema created successfully - no field naming errors!")
    
    # Start the server
    import uvicorn
    print("🎯 Starting server on http://localhost:8001")
    print("📊 GraphQL endpoint: http://localhost:8001/graphql")
    print("📚 API docs: http://localhost:8001/docs")
    print()
    print("🔬 Available clustering features:")
    print("   • K-Means clustering")
    print("   • Hierarchical clustering") 
    print("   • DBSCAN clustering")
    print("   • Candidate similarity search")
    print("   • Cluster analytics")
    print("   • GraphQL and REST APIs")
    print()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)