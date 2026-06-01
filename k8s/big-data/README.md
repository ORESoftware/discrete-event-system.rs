# Big Data Kubernetes Stack

This bundle deploys a small development-oriented big-data stack:

- Apache Spark standalone cluster: one master and two workers.
- Apache Airflow: one webserver/scheduler pod with a smoke-test DAG.
- MinIO: S3-compatible object storage for data-lake style inputs and outputs.

Databricks is not included because it is normally consumed as a managed service
or external control plane, not deployed as a plain in-cluster Kubernetes
`Deployment`.

## Apply

Authenticate to the Kubernetes cluster first, then apply the kustomization.
The configured non-local context in this workspace is currently
`factmachine-devnet`.

```bash
aws sso login
kubectl --context factmachine-devnet apply -k k8s/big-data
kubectl --context factmachine-devnet -n big-data get pods,svc
```

## Local UIs

```bash
kubectl --context factmachine-devnet -n big-data port-forward svc/spark-master 8080:8080
kubectl --context factmachine-devnet -n big-data port-forward svc/airflow 8082:8080
kubectl --context factmachine-devnet -n big-data port-forward svc/minio 9001:9001
```

- Spark UI: <http://localhost:8080>
- Airflow UI: <http://localhost:8082> (`admin` / `admin`)
- MinIO console: <http://localhost:9001> (`minioadmin` / `minioadmin123`)

## Notes

These manifests favor local development over production hardening:

- Airflow and MinIO use `emptyDir`, so data is ephemeral.
- Credentials are dev defaults and should be replaced before shared use.
- Airflow runs the webserver and scheduler in one pod.
- For production, prefer the official Airflow Helm chart, a Spark Operator or
  managed Spark platform, persistent volumes, external Postgres, and secret
  management.
