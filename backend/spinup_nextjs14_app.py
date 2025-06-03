import os
import subprocess
import sys
from pathlib import Path

def create_nextjs_app(project_name):
    print(f"🚀 Creating Next.js 14 application: {project_name}")
    subprocess.run([
        "npx", "create-next-app@latest", project_name,
        "--typescript",
        "--tailwind",
        "--eslint",
        "--app",
        "--src-dir",
        "--import-alias", "@/*",
        "--no-experimental-app"
    ], check=True)
    os.chdir(project_name)
    print("✅ Next.js app created successfully")

def install_dependencies():
    print("📦 Installing additional dependencies...")
    dependencies = [
        "zustand",
        "next-intl",
        "clsx",
        "tailwind-merge",
        "swr",
        "axios"
    ]
    subprocess.run(["npm", "install"] + dependencies, check=True)
    print("✅ Dependencies installed")

def setup_zustand():
    print("🛒 Setting up Zustand store...")
    stores_dir = Path("src/stores")
    stores_dir.mkdir(parents=True, exist_ok=True)
    
    store_content = """import { create } from 'zustand';

interface StoreState {
  // Your state properties here
}

interface StoreActions {
  // Your actions here
}

type Store = StoreState & StoreActions;

export const useStore = create<Store>((set) => ({
  // Initial state
  // Your state and actions implementation
}));

// Example usage:
// export const useSomeFeature = () => useStore(state => state.someFeature);
"""
    (stores_dir / "store.ts").write_text(store_content)
    print("✅ Zustand setup complete")

def setup_i18n():
    print("🌍 Setting up i18n with next-intl...")
    
    i18n_dir = Path("src/i18n")
    i18n_dir.mkdir(parents=True, exist_ok=True)
    
    config_content = """import {notFound} from 'next/navigation';
import {getRequestConfig} from 'next-intl/server';

const locales = ['en', 'fr'];

export default getRequestConfig(async ({locale}) => {
  if (!locales.includes(locale as any)) notFound();

  return {
    messages: (await import(`../messages/${locale}.json`)).default
  };
});
"""
    (i18n_dir / "i18n.ts").write_text(config_content)
    
    middleware_content = """import createMiddleware from 'next-intl/middleware';

export default createMiddleware({
  locales: ['en', 'fr'],
  defaultLocale: 'en'
});

export const config = {
  matcher: ['/', '/(fr|en)/:path*']
};
"""
    (Path("src") / "middleware.ts").write_text(middleware_content)
    
    next_config = Path("next.config.mjs")
    config_content = """/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: [],
  },
};

const withNextIntl = require('next-intl/plugin')(
  './src/i18n/i18n.ts'
);

module.exports = withNextIntl(nextConfig);
"""
    next_config.write_text(config_content)
    
    messages_dir = Path("src/messages")
    messages_dir.mkdir(parents=True, exist_ok=True)
    
    en_content = """{
  "Index": {
    "title": "Welcome to Next.js",
    "description": "Get started by editing src/app/page.tsx"
  }
}"""
    (messages_dir / "en.json").write_text(en_content)
    
    fr_content = """{
  "Index": {
    "title": "Bienvenue sur Next.js",
    "description": "Commencez par éditer src/app/page.tsx"
  }
}"""
    (messages_dir / "fr.json").write_text(fr_content)
    
    layout_path = Path("src/app/[locale]/layout.tsx")
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_content = """import { ReactNode } from 'react';
import { NextIntlClientProvider } from 'next-intl';
import { notFound } from 'next/navigation';

type Props = {
  children: ReactNode;
  params: { locale: string };
};

export default async function LocaleLayout({
  children,
  params: { locale }
}: Props) {
  let messages;
  try {
    messages = (await import(`../../messages/${locale}.json`)).default;
  } catch (error) {
    notFound();
  }

  return (
    <html lang={locale}>
      <body>
        <NextIntlClientProvider locale={locale} messages={messages}>
          {children}
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
"""
    layout_path.write_text(layout_content)
    
    page_path = Path("src/app/[locale]/page.tsx")
    page_content = """import { useTranslations } from 'next-intl';

export default function Index() {
  const t = useTranslations('Index');
  
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold">{t('title')}</h1>
      <p>{t('description')}</p>
    </div>
  );
}
"""
    page_path.write_text(page_content)
    
    root_layout_path = Path("src/app/layout.tsx")
    root_layout_content = """import { redirect } from 'next/navigation';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  redirect('/en');
}
"""
    root_layout_path.write_text(root_layout_content)
    
    print("✅ i18n setup complete")

def setup_data_services():
    print("📡 Setting up data services...")
    services_dir = Path("src/services")
    services_dir.mkdir(parents=True, exist_ok=True)
    
    base_service_content = """import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';

export abstract class BaseService {
  protected readonly http: AxiosInstance;
  private isRefreshing = false;
  private refreshSubscribers: ((token: string) => void)[] = [];

  protected constructor(baseURL: string, config?: AxiosRequestConfig) {
    this.http = axios.create({
      baseURL,
      ...config,
    });

    this.http.interceptors.request.use((config) => {
      const token = localStorage.getItem('accessToken');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    }, (error) => Promise.reject(error));

    this.http.interceptors.response.use(
      (response: AxiosResponse) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config;
        const status = error.response?.status;

        if (status === 401 || status === 403) {
          if (status === 401 && originalRequest.url?.includes('/refresh-token')) {
            this.clearAuth();
            window.location.href = '/login';
            return Promise.reject(error);
          }

          if (!this.isRefreshing) {
            this.isRefreshing = true;
            try {
              const newToken = await this.refreshToken();
              this.onRefreshed(newToken);
              return this.http(originalRequest);
            } catch (refreshError) {
              this.clearAuth();
              window.location.href = '/login';
              return Promise.reject(refreshError);
            } finally {
              this.isRefreshing = false;
            }
          }

          return new Promise((resolve, reject) => {
            this.subscribeTokenRefresh((token: string) => {
              originalRequest.headers.Authorization = `Bearer ${token}`;
              resolve(this.http(originalRequest));
            });
          });
        }

        return Promise.reject(error);
      }
    );
  }

  private subscribeTokenRefresh(callback: (token: string) => void) {
    this.refreshSubscribers.push(callback);
  }

  private onRefreshed(token: string) {
    this.refreshSubscribers.forEach(callback => callback(token));
    this.refreshSubscribers = [];
  }

  protected clearAuth() {
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
    document.cookie = 'token=; Path=/; Expires=Thu, 01 Jan 1970 00:00:01 GMT;';
  }

  protected async refreshToken(): Promise<string> {
    const refreshToken = localStorage.getItem('refreshToken');
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    try {
      const response = await axios.post('/api/auth/refresh-token', { refreshToken });
      const { accessToken, refreshToken: newRefreshToken } = response.data;

      localStorage.setItem('accessToken', accessToken);
      if (newRefreshToken) {
        localStorage.setItem('refreshToken', newRefreshToken);
      }

      return accessToken;
    } catch (error) {
      this.clearAuth();
      throw error;
    }
  }

  protected async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.http.get<T>(url, config);
    return response.data;
  }

  protected async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.http.post<T>(url, data, config);
    return response.data;
  }

  protected async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.http.put<T>(url, data, config);
    return response.data;
  }

  protected async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.http.delete<T>(url, config);
    return response.data;
  }
}
"""
    (services_dir / "BaseService.ts").write_text(base_service_content)
    
    api_config_content = """// API configuration constants
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:3000/api',
  TIMEOUT: 10000,
  HEADERS: {
    'Content-Type': 'application/json',
  },
  AUTH: {
    ACCESS_TOKEN_KEY: 'accessToken',
    REFRESH_TOKEN_KEY: 'refreshToken',
    TOKEN_REFRESH_ENDPOINT: '/auth/refresh-token',
    LOGIN_ENDPOINT: '/auth/login',
    LOGOUT_ENDPOINT: '/auth/logout',
    PROFILE_ENDPOINT: '/auth/profile',
  },
};

export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    LOGOUT: '/auth/logout',
    REFRESH: '/auth/refresh-token',
    PROFILE: '/auth/profile',
  },
};
"""
    (services_dir / "api.config.ts").write_text(api_config_content)
    
    auth_service_content = """import { BaseService } from './BaseService';
import { API_CONFIG, API_ENDPOINTS } from './api.config';

export class AuthService extends BaseService {
  private static instance: AuthService;

  private constructor() {
    super(API_CONFIG.BASE_URL, {
      timeout: API_CONFIG.TIMEOUT,
      headers: API_CONFIG.HEADERS,
    });
  }

  public static getInstance(): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService();
    }
    return AuthService.instance;
  }

  async login(credentials: { email: string; password: string }) {
    const response = await this.post<{
      accessToken: string;
      refreshToken: string;
      user: any;
    }>(API_ENDPOINTS.AUTH.LOGIN, credentials);

    if (response.accessToken && response.refreshToken) {
      localStorage.setItem(API_CONFIG.AUTH.ACCESS_TOKEN_KEY, response.accessToken);
      localStorage.setItem(API_CONFIG.AUTH.REFRESH_TOKEN_KEY, response.refreshToken);
    }

    return response;
  }

  async logout() {
    try {
      await this.post(API_ENDPOINTS.AUTH.LOGOUT);
    } finally {
      this.clearAuth();
    }
  }

  async getProfile() {
    return this.get(API_ENDPOINTS.AUTH.PROFILE);
  }
}

export const authService = AuthService.getInstance();
"""
    (services_dir / "AuthService.ts").write_text(auth_service_content)
    
    print("✅ Data services setup complete")

def setup_swr_hooks():
    print("🔄 Setting up SWR hooks...")
    hooks_dir = Path("src/hooks")
    hooks_dir.mkdir(parents=True, exist_ok=True)
    
    use_auth_content = """import { useEffect } from 'react';
import useSWR, { mutate } from 'swr';
import { authService } from '@/services/AuthService';
import { API_CONFIG } from '@/services/api.config';
import { useRouter } from 'next/navigation';

export function useAuth({ redirectIfUnauthenticated = '' } = {}) {
  const router = useRouter();
  
  const { data: user, error, isLoading, mutate: mutateUser } = useSWR(
    API_CONFIG.AUTH.PROFILE_ENDPOINT,
    async (url) => {
      try {
        return await authService.getProfile();
      } catch (error) {
        if (error.response?.status === 401) {
          try {
            const newToken = await authService.refreshToken();
            if (newToken) {
              return await authService.getProfile();
            }
          } catch (refreshError) {
            throw new Error('Authentication required');
          }
        }
        throw error;
      }
    },
    {
      revalidateOnFocus: false,
      shouldRetryOnError: false,
    }
  );

  useEffect(() => {
    if (error && redirectIfUnauthenticated) {
      router.push(redirectIfUnauthenticated);
    }
  }, [error, redirectIfUnauthenticated, router]);

  const login = async (credentials: { email: string; password: string }) => {
    const response = await authService.login(credentials);
    await mutateUser();
    return response;
  };

  const logout = async () => {
    await authService.logout();
    mutate(() => true, undefined, { revalidate: false });
    await mutateUser(null, false);
  };

  return {
    user,
    error,
    isLoading,
    login,
    logout,
    mutateUser,
    isAuthenticated: !!user && !error,
  };
}

export const fetcher = (url: string) => authService.get(url);

export function useProtectedSWR(key: string | null, config?: any) {
  const { user, error: authError } = useAuth();
  const swr = useSWR(key, fetcher, config);

  return {
    ...swr,
    error: authError || swr.error,
    isLoading: !user || swr.isLoading,
  };
}
"""
    (hooks_dir / "useAuth.ts").write_text(use_auth_content)
    
    print("✅ SWR hooks setup complete")

def install_shadcn():
    print("🎨 Installing shadcn/ui components...")
    
    subprocess.run(["npx", "shadcn-ui@latest", "init"], input=b"y\nsrc/components\n@/components\ny\ny\nslate\n", check=True)
    
    components = [
        "button",
        "input",
        "dropdown-menu",
        "alert",
        "card",
        "dialog",
        "form",
        "label",
        "toast",
        "sonner"
    ]
    
    for component in components:
        subprocess.run(["npx", "shadcn-ui@latest", "add", component], check=True)
    
    print("✅ shadcn/ui setup complete")

def main():
    if len(sys.argv) < 2:
        print("Usage: python setup_nextjs.py <project-name>")
        sys.exit(1)
    
    project_name = sys.argv[1]
    
    try:
        create_nextjs_app(project_name)
        install_dependencies()
        setup_zustand()
        setup_i18n()
        setup_data_services()
        setup_swr_hooks()
        install_shadcn()

        print(f"\n🎉 {project_name} Project setup completed successfully!")
        print(f"👉 Run the following commands to start:")
        print(f"cd {project_name}")
        print("npm run dev")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()