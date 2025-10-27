import strawberry
from app.graphql.queries import Query
from app.graphql.mutations import Mutation


# Create the GraphQL schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation
)