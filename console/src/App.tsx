import { Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./auth/AuthContext";
import { Layout } from "./components/Layout";
import { RequirePermission } from "./components/RequirePermission";
import { LoginPage } from "./pages/LoginPage";
import { FleetView } from "./pages/FleetView";
import { LiveSessionView } from "./pages/LiveSessionView";
import { ReviewWorkspace } from "./pages/ReviewWorkspace";
import { DatasetManagement } from "./pages/DatasetManagement";
import { TrainingVisibility } from "./pages/TrainingVisibility";
import { ModelRegistry } from "./pages/ModelRegistry";
import { KnowledgeBaseEditor } from "./pages/KnowledgeBaseEditor";
import { LogsAlerts } from "./pages/LogsAlerts";
import type { ReactNode } from "react";
import { useLocation } from "react-router-dom";

function RequireAuth({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const location = useLocation();
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  return <>{children}</>;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        element={
          <RequireAuth>
            <Layout />
          </RequireAuth>
        }
      >
        <Route index element={<Navigate to="/fleet" replace />} />
        <Route
          path="/fleet"
          element={
            <RequirePermission permission="fleet:view">
              <FleetView />
            </RequirePermission>
          }
        />
        <Route
          path="/sessions"
          element={
            <RequirePermission permission="session:view">
              <LiveSessionView />
            </RequirePermission>
          }
        />
        <Route
          path="/review"
          element={
            <RequirePermission permission="review:decide">
              <ReviewWorkspace />
            </RequirePermission>
          }
        />
        <Route
          path="/dataset"
          element={
            <RequirePermission permission="dataset:view">
              <DatasetManagement />
            </RequirePermission>
          }
        />
        <Route
          path="/training"
          element={
            <RequirePermission permission="training:view">
              <TrainingVisibility />
            </RequirePermission>
          }
        />
        <Route
          path="/registry"
          element={
            <RequirePermission permission="registry:view">
              <ModelRegistry />
            </RequirePermission>
          }
        />
        <Route
          path="/knowledge-base"
          element={
            <RequirePermission permission="kb:view">
              <KnowledgeBaseEditor />
            </RequirePermission>
          }
        />
        <Route
          path="/logs"
          element={
            <RequirePermission permission="logs:view">
              <LogsAlerts />
            </RequirePermission>
          }
        />
        <Route path="*" element={<Navigate to="/fleet" replace />} />
      </Route>
    </Routes>
  );
}

function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
}

export default App;
