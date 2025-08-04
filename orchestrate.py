#!/usr/bin/env python3
"""
Orchestration Management Script for YouTube Comment Intelligence
Provides easy commands to manage Docker Compose, Kubernetes, and monitoring
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

class OrchestrationManager:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.docker_compose_file = self.project_root / "docker-compose.yml"
        self.k8s_base_dir = self.project_root / "k8s" / "base"
        
    def run_command(self, command, description=""):
        """Run a command and handle errors."""
        if description:
            print(f"🚀 {description}")
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {description or 'Command executed successfully'}")
                return result.stdout
            else:
                print(f"❌ Error: {result.stderr}")
                return None
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None

    def check_docker(self):
        """Check if Docker is running."""
        return self.run_command("docker --version", "Checking Docker installation")

    def check_kubectl(self):
        """Check if kubectl is available."""
        return self.run_command("kubectl version --client", "Checking kubectl installation")

    def docker_compose_up(self, services=None):
        """Start Docker Compose services."""
        if services:
            cmd = f"docker-compose up -d {' '.join(services)}"
            return self.run_command(cmd, f"Starting services: {', '.join(services)}")
        else:
            return self.run_command("docker-compose up -d", "Starting all Docker Compose services")

    def docker_compose_down(self):
        """Stop Docker Compose services."""
        return self.run_command("docker-compose down", "Stopping Docker Compose services")

    def docker_compose_logs(self, service=None):
        """View Docker Compose logs."""
        if service:
            cmd = f"docker-compose logs -f {service}"
            return self.run_command(cmd, f"Viewing logs for {service}")
        else:
            return self.run_command("docker-compose logs -f", "Viewing all logs")

    def docker_compose_ps(self):
        """Show Docker Compose service status."""
        return self.run_command("docker-compose ps", "Service status")

    def k8s_deploy(self):
        """Deploy to Kubernetes."""
        if not self.k8s_base_dir.exists():
            print("❌ Kubernetes manifests not found")
            return None
        
        return self.run_command(f"kubectl apply -f {self.k8s_base_dir}/", "Deploying to Kubernetes")

    def k8s_status(self):
        """Check Kubernetes deployment status."""
        return self.run_command("kubectl get pods -n youtube-comment-intelligence", "Kubernetes deployment status")

    def k8s_logs(self, deployment):
        """View Kubernetes logs."""
        cmd = f"kubectl logs -f deployment/{deployment} -n youtube-comment-intelligence"
        return self.run_command(cmd, f"Viewing logs for {deployment}")

    def health_check(self):
        """Perform health checks on all services."""
        print("🏥 Performing health checks...")
        
        # API Health
        api_health = self.run_command("curl -s http://localhost:8080/health", "API Health Check")
        
        # Web UI Health
        web_health = self.run_command("curl -s http://localhost:8501", "Web UI Health Check")
        
        # Database Health
        db_health = self.run_command("docker-compose exec -T postgres pg_isready -U youtube_user", "Database Health Check")
        
        # Redis Health
        redis_health = self.run_command("docker-compose exec -T redis redis-cli ping", "Redis Health Check")
        
        print("✅ Health checks completed")

    def show_access_points(self):
        """Show all access points."""
        print("📍 Access Points:")
        print("   • Flask API: http://localhost:8080")
        print("   • Streamlit UI: http://localhost:8501")
        print("   • API Docs: http://localhost:8080/docs")
        print("   • Health Check: http://localhost:8080/health")
        print("   • Grafana: http://localhost:3000 (admin/admin)")
        print("   • Prometheus: http://localhost:9090")
        print("   • Kibana: http://localhost:5601")
        print("   • Flower: http://localhost:5555")
        print("   • Nginx: http://localhost:80")

    def scale_services(self, service, replicas):
        """Scale Docker Compose services."""
        cmd = f"docker-compose up -d --scale {service}={replicas}"
        return self.run_command(cmd, f"Scaling {service} to {replicas} replicas")

    def k8s_scale(self, deployment, replicas):
        """Scale Kubernetes deployments."""
        cmd = f"kubectl scale deployment {deployment} --replicas={replicas} -n youtube-comment-intelligence"
        return self.run_command(cmd, f"Scaling {deployment} to {replicas} replicas")

    def backup_data(self):
        """Create backup of application data."""
        print("💾 Creating backup...")
        
        # Create backup directory
        backup_dir = self.project_root / "backups" / time.strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup PostgreSQL
        self.run_command(f"docker-compose exec -T postgres pg_dump -U youtube_user youtube_intelligence > {backup_dir}/database.sql", "Backing up database")
        
        # Backup Redis
        self.run_command(f"docker-compose exec redis redis-cli BGSAVE", "Backing up Redis")
        
        print(f"✅ Backup created in {backup_dir}")

    def restore_data(self, backup_path):
        """Restore data from backup."""
        print(f"🔄 Restoring from {backup_path}...")
        
        if Path(backup_path).exists():
            self.run_command(f"docker-compose exec -T postgres psql -U youtube_user youtube_intelligence < {backup_path}/database.sql", "Restoring database")
            print("✅ Data restored successfully")
        else:
            print("❌ Backup not found")

    def show_metrics(self):
        """Show basic metrics."""
        print("📊 System Metrics:")
        
        # Container status
        self.run_command("docker-compose ps", "Container Status")
        
        # Resource usage
        self.run_command("docker stats --no-stream", "Resource Usage")
        
        # API metrics
        api_metrics = self.run_command("curl -s http://localhost:8080/metrics", "API Metrics")
        if api_metrics:
            print("✅ API metrics available")

def main():
    parser = argparse.ArgumentParser(description="YouTube Comment Intelligence Orchestration Manager")
    parser.add_argument("command", choices=[
        "start", "stop", "restart", "status", "logs", "health", "access", 
        "deploy", "scale", "backup", "restore", "metrics"
    ], help="Command to execute")
    
    parser.add_argument("--service", help="Specific service name")
    parser.add_argument("--replicas", type=int, help="Number of replicas for scaling")
    parser.add_argument("--backup-path", help="Path to backup for restore")
    
    args = parser.parse_args()
    
    manager = OrchestrationManager()
    
    if args.command == "start":
        manager.docker_compose_up(args.service)
        manager.show_access_points()
        
    elif args.command == "stop":
        manager.docker_compose_down()
        
    elif args.command == "restart":
        manager.docker_compose_down()
        time.sleep(2)
        manager.docker_compose_up()
        
    elif args.command == "status":
        manager.docker_compose_ps()
        
    elif args.command == "logs":
        manager.docker_compose_logs(args.service)
        
    elif args.command == "health":
        manager.health_check()
        
    elif args.command == "access":
        manager.show_access_points()
        
    elif args.command == "deploy":
        manager.k8s_deploy()
        manager.k8s_status()
        
    elif args.command == "scale":
        if not args.service or not args.replicas:
            print("❌ Please specify --service and --replicas")
            return
        manager.scale_services(args.service, args.replicas)
        
    elif args.command == "backup":
        manager.backup_data()
        
    elif args.command == "restore":
        if not args.backup_path:
            print("❌ Please specify --backup-path")
            return
        manager.restore_data(args.backup_path)
        
    elif args.command == "metrics":
        manager.show_metrics()

if __name__ == "__main__":
    main() 